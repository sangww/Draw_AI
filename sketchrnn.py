import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

class HParams():
    def __init__(self):
        self.input_dim = 2
        self.style_dim = 3
        self.enc_hidden_size = 64  # 256
        self.dec_hidden_size = 128  # 512
        self.Nz = 16   # latent dimension
        self.M = 3   # 20 # gaussian mixture
        self.dropout = 0.0  # 0.9
        #self.batch_size = 16
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.lr = 0.001
        self.lr_decay = 0.999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.4
        self.max_seq_length = 200
        self.Nmax = 100

hp = HParams()

def lr_decay(optimizer):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group['lr']>hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer

class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.lstm = nn.LSTM(hp.input_dim, hp.enc_hidden_size, dropout=hp.dropout, bidirectional=True)
        self.fc_mu = nn.Linear(2*hp.enc_hidden_size + hp.style_dim, hp.Nz)
        self.fc_sigma = nn.Linear(2*hp.enc_hidden_size + hp.style_dim, hp.Nz)

        self.train()

    def forward(self, inputs, labels, hidden_cell=None):
        batch_size = inputs.size(1) # input is L N C
        if hidden_cell is None:
            hidden = torch.zeros(2, batch_size, hp.enc_hidden_size).cuda()
            cell = torch.zeros(2, batch_size, hp.enc_hidden_size).cuda()
            hidden_cell = (hidden, cell)
        _, (hidden, cell) = self.lstm(inputs, hidden_cell)
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = torch.split(hidden,1,0)
        hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0), labels],1)

        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat / 2.0)

        z_size = mu.size()
        N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda()
        z = mu + sigma * N

        #self.mu = mu
        #self.sigma_hat = sigma_hat
        #self.sigma = sigma

        return z, mu, sigma_hat

class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        # to init hidden and cell from z:
        self.fc_hc = nn.Linear(hp.Nz + hp.style_dim, 2*hp.dec_hidden_size)
        # unidirectional lstm:
        self.lstm = nn.LSTM(hp.Nz+hp.input_dim, hp.dec_hidden_size, dropout=hp.dropout)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size,6*hp.M) # no pen state for now...

    def forward(self, inputs, z, labels, hidden_cell=None):
        #print ("decoder inputs", inputs.size())
        z = torch.cat([z, labels], dim=1)
        if hidden_cell is None:
            # then we must init from z
            hidden,cell = torch.split(torch.tanh(self.fc_hc(z)),hp.dec_hidden_size,1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs,(hidden,cell) = self.lstm(inputs, hidden_cell)

        #print ("decoder outputs", outputs.size())

        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        if self.training:
            y = self.fc_params(outputs.view(-1, hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, hp.dec_hidden_size))

        #print ("y", y.size())

        params = torch.split(y,6,1)
        #for i, item in enumerate(params):
        #    print (i, item.size())
        params_mixture = torch.stack(params[:]) # trajectory
        #print("params mixture", params_mixture.size())
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)

        #for item in [pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy]:
        #    print (item.size())

        if self.training:
            len_out = hp.Nmax + 1
        else:
            len_out = 1

        pi = F.softmax(pi.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(len_out,-1,hp.M)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(len_out,-1,hp.M)

        return pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy,hidden,cell

class SketchRNN():
    def __init__(self):
        self.encoder = EncoderRNN().cuda()
        self.decoder = DecoderRNN().cuda()

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min

    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx-self.mu_x)/self.sigma_x)**2
        z_y = ((dy-self.mu_y)/self.sigma_y)**2
        z_xy = (dx-self.mu_x)*(dy-self.mu_y)/(self.sigma_x*self.sigma_y)
        z = z_x + z_y -2*self.rho_xy*z_xy
        exp = torch.exp(-z/(2*(1-self.rho_xy**2)))
        norm = 2*np.pi*self.sigma_x*self.sigma_y*torch.sqrt(1-self.rho_xy**2)
        return exp / norm

    def reconstruction_loss(self, mask, dx, dy, batch_size):
        pdf = self.bivariate_normal_pdf(dx, dy)
        LS = -torch.sum(mask * torch.log(1e-5 + torch.sum(self.pi * pdf, 2)))\
            / float(hp.Nmax * batch_size)
        # remove pen state for now
        #LP = -torch.sum(p*torch.log(self.q))/float(Nmax*hp.batch_size)
        return LS

    def KL_loss(self, batch_size):
        LKL = -0.5*torch.sum(1+self.sigma-self.mu**2-torch.exp(self.sigma))\
            / float(hp.Nz * batch_size)
        KL_min = Variable(torch.Tensor([hp.KL_min]).cuda()).detach()
        return hp.wKL*self.eta_step * torch.max(LKL,KL_min)

    # assume equal lengths
    def make_target(self, batch):
        batch_size = batch.size(1)
        eos = torch.stack([torch.Tensor([0.0, 0.0])] * batch_size).cuda().unsqueeze(0)
        batch = torch.cat([batch, eos], 0)

        mask = torch.ones(hp.Nmax + 1, batch_size)
        mask[hp.Nmax, :] = 0.0
        mask = mask.cuda()

        dx = torch.stack([batch.data[:, :, 0]] * hp.M, 2)
        dy = torch.stack([batch.data[:, :, 1]] * hp.M, 2)

        return mask, dx, dy

    def train(self, dataloader, epoch):
        self.encoder.train()
        self.decoder.train()

        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            batch_size = inputs.size(0)

            # N C L -> L N C
            inputs = inputs.permute(2, 0, 1)

            #assert batch_size == hp.batch_size
            assert batch_size == inputs.size(1)
            assert hp.Nmax == inputs.size(0)
            # sigma is actually sigma_hat
            z, self.mu, self.sigma = self.encoder(inputs, labels)

            sos = torch.stack([torch.Tensor([0.0, 0.0])] * batch_size).cuda().unsqueeze(0)

            decoder_inputs = torch.cat([sos, inputs], 0)

            z_stack = torch.stack([z] * (hp.Nmax+1))

            # decoder concatenates sequence and z at every time step
            decoder_inputs = torch.cat([decoder_inputs, z_stack], 2)

            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, \
                hidden, cell = self.decoder(decoder_inputs, z, labels)

            mask, dx, dy = self.make_target(inputs)

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            self.eta_step = 1.0 # 1.0 - (1.0 - hp.eta_min) * hp.R

            L1 = self.KL_loss(batch_size)
            L2 = self.reconstruction_loss(mask, dx, dy, batch_size)
            loss = L1 + L2
            loss.backward()

            nn.utils.clip_grad_norm(self.encoder.parameters(), hp.grad_clip)
            nn.utils.clip_grad_norm(self.decoder.parameters(), hp.grad_clip)

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

        print("Epoch", epoch, "Loss KL", L1.item(), "Loss R", L2.item())
        self.encoder_optimizer = lr_decay(self.encoder_optimizer)
        self.decoder_optimizer = lr_decay(self.decoder_optimizer)

        if epoch > 0 and epoch % 20 == 0:
            self.save(epoch)

    def conditional_generation(self, inputs, labels, greedy=False):
        # should remove dropouts:
        self.encoder.train(False)
        self.decoder.train(False)

        # L N C
        batch_size = inputs.size(1)
        assert batch_size == 1
        assert batch_size == labels.size(0)

        # encode:
        z, _, _ = self.encoder(inputs, labels)

        sos = Variable(torch.Tensor([0.0, 0.0]).view(1,1,-1).cuda())

        s = sos
        seq_x = []
        seq_y = []
        #seq_z = []
        hidden_cell = None
        for i in range(hp.Nmax):
            decoder_inputs = torch.cat([s,z.unsqueeze(0)], 2)

            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, hidden, cell = \
                    self.decoder(decoder_inputs, z, labels, hidden_cell)
            hidden_cell = (hidden, cell)
            # sample from parameters:
            #s, dx, dy, pen_down, eos = self.sample_next_state()
            s, dx, dy = self.sample_next_state(greedy)
            #------
            seq_x.append(dx)
            seq_y.append(dy)
            #seq_z.append(pen_down)
        # visualize result:

        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        #z_sample = np.array(seq_z)
        return x_sample, y_sample, seq_x, seq_y

    def sample_next_state(self, greedy=False):

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf) / hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0,0,:].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(hp.M, p=pi)
        #print (pi_idx)
        # get pen state:
        #q = self.q.data[0,0,:].cpu().numpy()
        #q = adjust_temp(q)
        #q_idx = np.random.choice(3, p=q)
        # get mixture params:
        mu_x = self.mu_x.data[0,0,pi_idx].item()
        mu_y = self.mu_y.data[0,0,pi_idx].item()
        sigma_x = self.sigma_x.data[0,0,pi_idx].item()
        sigma_y = self.sigma_y.data[0,0,pi_idx].item()
        rho_xy = self.rho_xy.data[0,0,pi_idx].item()
        x,y = sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy,greedy=greedy)
        next_state = torch.zeros(2) ## temp
        next_state[0] = x
        next_state[1] = y
        #next_state[q_idx+2] = 1
        return Variable(next_state.cuda()).view(1,1,-1), x, y

    def save(self, epoch):
        sel = np.random.rand()
        torch.save(self.encoder.state_dict(), \
            'encoderRNN_sel_%3f_epoch_%d.pth' % (sel,epoch))
        torch.save(self.decoder.state_dict(), \
            'decoderRNN_sel_%3f_epoch_%d.pth' % (sel,epoch))

def sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy, greedy=False):
    # inputs must be floats
    if greedy:
      return mu_x,mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(hp.temperature)
    sigma_y *= np.sqrt(hp.temperature)
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],\
        [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]