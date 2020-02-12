from sketch_transfer import *

class SketchTransfer_1enc():
    def __init__(self, hp):
        self.hp = hp
        self.encoder = BiLSTMEncoder(hp, True).cuda()
        self.decoder = LSTMDecoder(hp, False).cuda()

        self.optim_encoder = optim.Adam(self.encoder.parameters(), hp.lr)
        self.optim_decoder = optim.Adam(self.decoder.parameters(), hp.lr)
        self.KL_weight = hp.KL_start

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
            / float(self.hp.Nmax * batch_size)
        # remove pen state for now
        #LP = -torch.sum(p*torch.log(self.q))/float(Nmax*hp.batch_size)
        return LS
    
    def KL_loss(self, batch_size):
        LKL = -0.5*torch.sum(1+self.sigma_stroke - self.mu_stroke**2-torch.exp(self.sigma_stroke))\
            / float(self.hp.Nz * batch_size)
        KL_min = Variable(torch.Tensor([self.hp.KL_min]).cuda()).detach()
        #return hp.wKL*self.eta_step * torch.max(LKL,KL_min)
        return self.hp.wKL * self.KL_weight * torch.max(LKL,KL_min)

    # assume equal lengths
    def make_target(self, batch):
        batch_size = batch.size(1)
        eos = torch.stack([torch.Tensor([0.0, 0.0])] * batch_size).cuda().unsqueeze(0)
        batch = torch.cat([batch, eos], 0)

        mask = torch.ones(self.hp.Nmax + 1, batch_size)
        mask[self.hp.Nmax, :] = 0.0
        mask = mask.cuda()

        dx = torch.stack([batch.data[:, :, 0]] * self.hp.M, 2)
        dy = torch.stack([batch.data[:, :, 1]] * self.hp.M, 2)

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

            assert batch_size == inputs.size(1)
            assert self.hp.Nmax == inputs.size(0)

            stroke = inputs[:, :, :2]
            z, self.mu_stroke, self.sigma_stroke = self.encoder(stroke, labels)

            sos = torch.stack([torch.Tensor([0.0, 0.0])] * batch_size).cuda().unsqueeze(0)

            decoder_inputs = torch.cat([sos, stroke], 0)

            z_stack = torch.stack([z] * (self.hp.Nmax+1))

            # decoder concatenates sequence and z at every time step
            decoder_inputs = torch.cat([decoder_inputs, z_stack], 2)

            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, \
                hidden, cell = self.decoder(decoder_inputs, z)

            mask, dx, dy = self.make_target(stroke)

            self.optim_encoder.zero_grad()
            self.optim_decoder.zero_grad()

            L_KL = self.KL_loss(batch_size)
            L_R = self.reconstruction_loss(mask, dx, dy, batch_size)
            loss = L_KL + L_R
            loss.backward()

            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.hp.grad_clip)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.hp.grad_clip)

            self.optim_encoder.step()
            self.optim_decoder.step()

        print("Epoch", epoch, "Loss KL", L_KL.item(), "Loss R", L_R.item())
        self.optim_encoder = lr_decay(self.optim_encoder)
        self.optim_decoder = lr_decay(self.optim_decoder)
        
        if self.KL_weight < 1.0:
            self.KL_weight += self.hp.KL_delta

        if epoch > 0 and epoch % self.hp.save_every == 0:
            self.save(epoch)

    def test_reconstruction(self, inputs, labels, greedy=False):
        self.encoder.train(False)
        self.decoder.train(False)

        # L N C
        batch_size = inputs.size(1)
        assert batch_size == 1
        assert batch_size == labels.size(0)

        # Encode
        stroke = inputs[:, :, :2]
        z, _, __ = self.encoder(stroke, labels)

        sos = Variable(torch.Tensor([0.0, 0.0]).view(1,1,-1).cuda())
        s = sos
        seq_x = []
        seq_y = []
        hidden_cell = None
        for i in range(self.hp.Nmax):
            decoder_inputs = torch.cat([s, z.unsqueeze(0)], 2)

            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, hidden, cell = \
                    self.decoder(decoder_inputs, z, None, hidden_cell)
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

    def generate_with_latent(self, stroke_latent, steps=None, last=None, hidden_cell=None, greedy=False):
        self.encoder.train(False)
        self.decoder.train(False)

        # L N C
        batch_size = stroke_latent.size(0)
        assert batch_size == 1

        if not steps:
            steps = self.hp.Nmax
        #assert steps <= self.hp.Nmax
        #z = torch.cat((z1, stroke_latent), dim=1)
        z = stroke_latent

        sos = Variable(torch.Tensor([0.0, 0.0]).view(1,1,-1).cuda())
        s = sos
        seq_x = []
        seq_y = []
        for i in range(steps):
            decoder_inputs = torch.cat([s, z.unsqueeze(0)], 2)
            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, hidden, cell = \
                    self.decoder(decoder_inputs, z, None, hidden_cell)
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
            pi_pdf = np.log(pi_pdf) / self.hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0,0,:].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(self.hp.M, p=pi)
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
            'sketch_encoder_sel_%3f_epoch_%d.pth' % (sel,epoch))
        torch.save(self.decoder.state_dict(), \
            'sketch_decoder_sel_%3f_epoch_%d.pth' % (sel,epoch))

class SketchTransfer_enc_dec():
    def __init__(self, hp):
        self.hp = hp
        self.encoder = BiLSTMEncoder(hp, True).cuda()
        self.decoder = LSTMDecoder(hp, True).cuda()

        self.optim_encoder = optim.Adam(self.encoder.parameters(), hp.lr)
        self.optim_decoder = optim.Adam(self.decoder.parameters(), hp.lr)
        self.KL_weight = hp.KL_start

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
            / float(self.hp.Nmax * batch_size)
        # remove pen state for now
        #LP = -torch.sum(p*torch.log(self.q))/float(Nmax*hp.batch_size)
        return LS
    
    def KL_loss(self, batch_size):
        LKL = -0.5*torch.sum(1+self.sigma_stroke - self.mu_stroke**2-torch.exp(self.sigma_stroke))\
            / float(self.hp.Nz * batch_size)
        KL_min = Variable(torch.Tensor([self.hp.KL_min]).cuda()).detach()
        #return hp.wKL*self.eta_step * torch.max(LKL,KL_min)
        return self.hp.wKL * self.KL_weight * torch.max(LKL,KL_min)

    # assume equal lengths
    def make_target(self, batch):
        batch_size = batch.size(1)
        eos = torch.stack([torch.Tensor([0.0, 0.0])] * batch_size).cuda().unsqueeze(0)
        batch = torch.cat([batch, eos], 0)

        mask = torch.ones(self.hp.Nmax + 1, batch_size)
        mask[self.hp.Nmax, :] = 0.0
        mask = mask.cuda()

        dx = torch.stack([batch.data[:, :, 0]] * self.hp.M, 2)
        dy = torch.stack([batch.data[:, :, 1]] * self.hp.M, 2)

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

            assert batch_size == inputs.size(1)
            assert self.hp.Nmax == inputs.size(0)

            stroke = inputs[:, :, :2]
            z, self.mu_stroke, self.sigma_stroke = self.encoder(stroke, labels)

            z = torch.cat([z, labels], dim=1) # concatenate latent vectors with style labels

            sos = torch.stack([torch.Tensor([0.0, 0.0])] * batch_size).cuda().unsqueeze(0)

            decoder_inputs = torch.cat([sos, stroke], 0)

            z_stack = torch.stack([z] * (self.hp.Nmax+1))

            # decoder concatenates sequence and z at every time step
            decoder_inputs = torch.cat([decoder_inputs, z_stack], 2)

            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, \
                hidden, cell = self.decoder(decoder_inputs, z)

            mask, dx, dy = self.make_target(stroke)

            self.optim_encoder.zero_grad()
            self.optim_decoder.zero_grad()

            L_KL = self.KL_loss(batch_size)
            L_R = self.reconstruction_loss(mask, dx, dy, batch_size)
            loss = L_KL + L_R
            loss.backward()

            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.hp.grad_clip)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.hp.grad_clip)

            self.optim_encoder.step()
            self.optim_decoder.step()

        print("Epoch", epoch, "Loss KL", L_KL.item(), "Loss R", L_R.item())
        self.optim_encoder = lr_decay(self.optim_encoder)
        self.optim_decoder = lr_decay(self.optim_decoder)
        
        if self.KL_weight < 1.0:
            self.KL_weight += self.hp.KL_delta

        if epoch > 0 and epoch % self.hp.save_every == 0:
            self.save(epoch)

    def test_reconstruction(self, inputs, labels, greedy=False):
        self.encoder.train(False)
        self.decoder.train(False)

        # L N C
        batch_size = inputs.size(1)
        assert batch_size == 1
        assert batch_size == labels.size(0)

        # Encode
        stroke = inputs[:, :, :2]
        z, _, __ = self.encoder(stroke, labels)

        z = torch.cat([z, labels], dim=1)

        sos = Variable(torch.Tensor([0.0, 0.0]).view(1,1,-1).cuda())
        s = sos
        seq_x = []
        seq_y = []
        hidden_cell = None
        for i in range(self.hp.Nmax):
            decoder_inputs = torch.cat([s, z.unsqueeze(0)], 2)

            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, hidden, cell = \
                    self.decoder(decoder_inputs, z, None, hidden_cell)
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

    def generate_with_latent(self, stroke_latent, labels, steps=None, last=None, hidden_cell=None, greedy=False):
        self.encoder.train(False)
        self.decoder.train(False)

        # L N C
        batch_size = stroke_latent.size(0)
        assert batch_size == 1

        if not steps:
            steps = self.hp.Nmax
        #assert steps <= self.hp.Nmax

        #last_L = 0
        #if last:
        #    last_L = last.size(0)

        #z1, _, __ = self.encoder_control(control)
        #print(z1)
        #z = torch.cat((z1, stroke_latent), dim=1)
        z = stroke_latent

        z = torch.cat([z, labels], dim=1)

        sos = Variable(torch.Tensor([0.0, 0.0]).view(1,1,-1).cuda())
        s = sos
        seq_x = []
        seq_y = []
        #hidden_cell = None
        for i in range(steps):
            decoder_inputs = torch.cat([s, z.unsqueeze(0)], 2)

            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, hidden, cell = \
                    self.decoder(decoder_inputs, z, None, hidden_cell)
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
            pi_pdf = np.log(pi_pdf) / self.hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0,0,:].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(self.hp.M, p=pi)
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
            'sketch_encoder_sel_%3f_epoch_%d.pth' % (sel,epoch))
        torch.save(self.decoder.state_dict(), \
            'sketch_decoder_sel_%3f_epoch_%d.pth' % (sel,epoch))

class SketchTransfer_nolabel():
    def __init__(self, hp):
        self.hp = hp
        self.encoder = BiLSTMEncoder(hp, False).cuda()
        self.decoder = LSTMDecoder(hp, False).cuda()

        self.optim_encoder = optim.Adam(self.encoder.parameters(), hp.lr)
        self.optim_decoder = optim.Adam(self.decoder.parameters(), hp.lr)
        self.KL_weight = hp.KL_start

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
            / float(self.hp.Nmax * batch_size)
        # remove pen state for now
        #LP = -torch.sum(p*torch.log(self.q))/float(Nmax*hp.batch_size)
        return LS
    
    def KL_loss(self, batch_size):
        LKL = -0.5*torch.sum(1+self.sigma_stroke - self.mu_stroke**2-torch.exp(self.sigma_stroke))\
            / float(self.hp.Nz * batch_size)
        KL_min = Variable(torch.Tensor([self.hp.KL_min]).cuda()).detach()
        #return hp.wKL*self.eta_step * torch.max(LKL,KL_min)
        return self.hp.wKL * self.KL_weight * torch.max(LKL,KL_min)

    # assume equal lengths
    def make_target(self, batch):
        batch_size = batch.size(1)
        eos = torch.stack([torch.Tensor([0.0, 0.0])] * batch_size).cuda().unsqueeze(0)
        batch = torch.cat([batch, eos], 0)

        mask = torch.ones(self.hp.Nmax + 1, batch_size)
        mask[self.hp.Nmax, :] = 0.0
        mask = mask.cuda()

        dx = torch.stack([batch.data[:, :, 0]] * self.hp.M, 2)
        dy = torch.stack([batch.data[:, :, 1]] * self.hp.M, 2)

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

            assert batch_size == inputs.size(1)
            assert self.hp.Nmax == inputs.size(0)

            stroke = inputs[:, :, :2]
            z, self.mu_stroke, self.sigma_stroke = self.encoder(stroke, None)

            sos = torch.stack([torch.Tensor([0.0, 0.0])] * batch_size).cuda().unsqueeze(0)

            decoder_inputs = torch.cat([sos, stroke], 0)

            z_stack = torch.stack([z] * (self.hp.Nmax+1))

            # decoder concatenates sequence and z at every time step
            decoder_inputs = torch.cat([decoder_inputs, z_stack], 2)

            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, \
                hidden, cell = self.decoder(decoder_inputs, z)

            mask, dx, dy = self.make_target(stroke)

            self.optim_encoder.zero_grad()
            self.optim_decoder.zero_grad()

            L_KL = self.KL_loss(batch_size)
            L_R = self.reconstruction_loss(mask, dx, dy, batch_size)
            loss = L_KL + L_R
            loss.backward()

            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.hp.grad_clip)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.hp.grad_clip)

            self.optim_encoder.step()
            self.optim_decoder.step()

        print("Epoch", epoch, "Loss KL", L_KL.item(), "Loss R", L_R.item())
        self.optim_encoder = lr_decay(self.optim_encoder)
        self.optim_decoder = lr_decay(self.optim_decoder)
        
        if self.KL_weight < 1.0:
            self.KL_weight += self.hp.KL_delta

        if epoch > 0 and epoch % self.hp.save_every == 0:
            self.save(epoch)

    def test_reconstruction(self, inputs, labels, greedy=False):
        self.encoder.train(False)
        self.decoder.train(False)

        # L N C
        batch_size = inputs.size(1)
        assert batch_size == 1
        assert batch_size == labels.size(0)

        # Encode
        stroke = inputs[:, :, :2]
        z, _, __ = self.encoder(stroke, None)

        sos = Variable(torch.Tensor([0.0, 0.0]).view(1,1,-1).cuda())
        s = sos
        seq_x = []
        seq_y = []
        hidden_cell = None
        for i in range(self.hp.Nmax):
            decoder_inputs = torch.cat([s, z.unsqueeze(0)], 2)

            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, hidden, cell = \
                    self.decoder(decoder_inputs, z, None, hidden_cell)
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

    def generate_with_latent(self, stroke_latent, steps=None, last=None, hidden_cell=None, greedy=False):
        self.encoder.train(False)
        self.decoder.train(False)

        # L N C
        batch_size = stroke_latent.size(0)
        assert batch_size == 1

        if not steps:
            steps = self.hp.Nmax

        z = stroke_latent

        sos = Variable(torch.Tensor([0.0, 0.0]).view(1,1,-1).cuda())
        s = sos
        seq_x = []
        seq_y = []
        #hidden_cell = None
        for i in range(steps):
            decoder_inputs = torch.cat([s, z.unsqueeze(0)], 2)

            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, hidden, cell = \
                    self.decoder(decoder_inputs, z, None, hidden_cell)
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

    def generate_sequence(self, latent, steps, ts_sequence=None, hidden_cell=None):
        self.decoder.train(False)
        batch_size = latent.size(0)
        assert batch_size == 1
        z = latent

        if ts_sequence is None:
            sos = Variable(torch.Tensor([0.0, 0.0]).view(1,1,-1).cuda())
            s = sos
        else:
            # L N C
            seq_len = ts_sequence.size(0)
            s = ts_sequence[seq_len-1, :, :].view(1,1,-1)

        seq_x = []
        seq_y = []
        for i in range(steps):
            decoder_inputs = torch.cat([s, z.unsqueeze(0)], 2)

            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, hidden, cell = \
                    self.decoder(decoder_inputs, z, None, hidden_cell)
            hidden_cell = (hidden, cell)
            
            s, dx, dy = self.sample_next_state(False)

            seq_x.append(dx)
            seq_y.append(dy)
        return s, seq_x, seq_y, hidden_cell

    def generate_full_sequence(self, latent, steps, ts_sequence=None, hidden_cell=None):
        self.decoder.train(False)
        batch_size = latent.size(0)
        assert batch_size == 1
        z = latent

        if ts_sequence is None:
            sos = Variable(torch.Tensor([0.0, 0.0]).view(1,1,-1).cuda())
            ts_sequence = sos

        seq_x = []
        seq_y = []
        for i in range(steps):
            seq_len = ts_sequence.size(0)
            z_stack = torch.stack([z] * (seq_len))
            decoder_inputs = torch.cat([ts_sequence, z_stack], 2)

            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, hidden, cell = \
                    self.decoder(decoder_inputs, z, None, hidden_cell)
            hidden_cell = (hidden, cell)
            
            s, dx, dy = self.sample_next_state(False)
            ts_sequence = torch.cat([ts_sequence, s], 0)

            seq_x.append(dx)
            seq_y.append(dy)
        return ts_sequence, seq_x, seq_y, hidden_cell

    def sample_next_state(self, greedy=False):

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf) / self.hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0,0,:].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(self.hp.M, p=pi)
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
            'sketch_encoder_sel_%3f_epoch_%d.pth' % (sel,epoch))
        torch.save(self.decoder.state_dict(), \
            'sketch_decoder_sel_%3f_epoch_%d.pth' % (sel,epoch))