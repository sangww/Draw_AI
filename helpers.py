import numpy as np

class HParams():
    def __init__(self):
        self.input_dim = 2
        self.style_dim = 3
        self.enc_hidden_size = 64  # 256
        self.dec_hidden_size = 128  # 512
        self.Nz = 2   # latent dimension
        self.Nz_dec = 4
        self.M = 3   # 20 # gaussian mixture
        self.dropout = 0.0  # 0.9
        #self.batch_size = 16
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.KL_a = 0.1
        self.KL_start = 0.01
        self.KL_delta = 0.01
        self.lr = 0.001
        self.lr_decay = 0.999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.4
        self.max_seq_length = 200
        self.Nmax = 100
        self.save_every = 20

hp = HParams()

def lr_decay(optimizer):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group['lr']>hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer

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