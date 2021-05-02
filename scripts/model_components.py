import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from helpers import *


class BiLSTMEncoder(nn.Module):
    def __init__(self, hp, style_label=False):
        super(BiLSTMEncoder, self).__init__()
        self.hp = hp
        self.lstm = nn.LSTM(hp.input_dim, hp.enc_hidden_size, hp.enc_layers, dropout=hp.dropout, bidirectional=True)
        if style_label:
            self.fc_mu = nn.Linear(2*hp.enc_layers*hp.enc_hidden_size + hp.style_dim, hp.Nz)
            self.fc_sigma = nn.Linear(2*hp.enc_layers*hp.enc_hidden_size + hp.style_dim, hp.Nz)
        else:
            self.fc_mu = nn.Linear(2*hp.enc_layers*hp.enc_hidden_size, hp.Nz)
            self.fc_sigma = nn.Linear(2*hp.enc_layers*hp.enc_hidden_size, hp.Nz)

        self.train()

    def forward(self, inputs, labels=None, hidden_cell=None):
        batch_size = inputs.size(1) # input is L N C
        if hidden_cell is None:
            hidden = torch.zeros(self.hp.enc_layers*2, batch_size, self.hp.enc_hidden_size).cuda()
            cell = torch.zeros(self.hp.enc_layers*2, batch_size, self.hp.enc_hidden_size).cuda()
            hidden_cell = (hidden, cell)
        _, (hidden, cell) = self.lstm(inputs, hidden_cell)
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = torch.split(hidden,self.hp.enc_layers,0)
        hidden_forward = hidden_forward.permute(1, 0, 2).reshape((batch_size, -1))
        hidden_backward = hidden_backward.permute(1, 0, 2).reshape((batch_size, -1))
        if labels is None:
            hidden_cat = torch.cat([hidden_forward, hidden_backward],1)
        else:
            hidden_cat = torch.cat([hidden_forward, hidden_backward, labels],1)

        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat / 2.0)

        z_size = mu.size()
        N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda()
        z = mu + sigma * N

        return z, mu, sigma_hat

class LSTMDecoder(nn.Module):
    def __init__(self, hp, style_label=False):
        super(LSTMDecoder, self).__init__()
        self.hp = hp
        # to init hidden and cell from z:
        if style_label:
            self.fc_hc = nn.Linear(hp.Nz_dec + hp.style_dim, 2*hp.dec_hidden_size*hp.dec_layers)
            self.lstm = nn.LSTM(hp.Nz_dec + hp.style_dim + hp.input_dim, hp.dec_hidden_size, hp.dec_layers, dropout=hp.dropout)
        else:
            self.fc_hc = nn.Linear(hp.Nz_dec, 2*hp.dec_hidden_size*hp.dec_layers)
            self.lstm = nn.LSTM(hp.Nz_dec + hp.input_dim, hp.dec_hidden_size, hp.dec_layers, dropout=hp.dropout)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6*hp.M) # no pen state for now...

    def forward(self, inputs, z, labels=None, hidden_cell=None):
        if labels is not None:
            z = torch.cat([z, labels], dim=1)
        if hidden_cell is None:
            # then we must init from z
            hidden,cell = torch.split(torch.tanh(self.fc_hc(z)), self.hp.dec_hidden_size * self.hp.dec_layers, 1)
            if self.hp.dec_layers == 1:
                hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
            else:
                batch_size = hidden.size(0)
                hidden = hidden.reshape((batch_size, self.hp.dec_layers, -1)).permute(1, 0, 2).contiguous()
                cell = cell.reshape((batch_size, self.hp.dec_layers, -1)).permute(1, 0, 2).contiguous()
                hidden_cell = (hidden, cell)

                #hidden = torch.zeros(self.hp.enc_layers*2, batch_size, self.hp.enc_hidden_size).cuda()
                #cell = torch.zeros(self.hp.enc_layers*2, batch_size, self.hp.enc_hidden_size).cuda()
        outputs,(hidden,cell) = self.lstm(inputs, hidden_cell)

        #print ("decoder outputs", outputs.size())

        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        #if self.training:
        y = self.fc_params(outputs.view(-1, self.hp.dec_hidden_size))
        #else:
        #    y = self.fc_params(hidden.view(-1, self.hp.dec_hidden_size*self.hp.dec_layers))


        params = torch.split(y,6,1)
        #for i, item in enumerate(params):
        #    print (i, item.size())
        params_mixture = torch.stack(params[:]) # trajectory
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)

        #for item in [pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy]:
        #    print (item.size())

        if self.training:
            len_out = self.hp.Nmax + 1
        else:
            len_out = 1

        pi = F.softmax(pi.transpose(0,1).squeeze()).view(len_out,-1,self.hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(len_out,-1,self.hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(len_out,-1,self.hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(len_out,-1,self.hp.M)
        mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(len_out,-1,self.hp.M)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(len_out,-1,self.hp.M)

        return pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy,hidden,cell