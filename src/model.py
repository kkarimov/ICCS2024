# Author: Karim Salta (Karimov)
# contacts: kkarimov@gmail.com, https://github.com/kkarimov

from src.OneToOneLinear import *
import torch
import torch.nn as nn
from torch.autograd import Variable


class OTOLayer(nn.Module):
    def __init__(self, n_input):
        super(OTOLayer, self).__init__()
        self.n_input = n_input
        self.OTO = OneToOneLinear(n_input)

    def forward(self, x):
        return self.OTO(x)


class FC_VAE(nn.Module):
    """Fully connected variational Autoencoder"""
    def __init__(self, n_input, nz, n_hidden=1024, useGPU = False, sparse = False):
        super(FC_VAE, self).__init__()
        self.nz = nz
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.useGPU = useGPU
        self.sparse = sparse

        self.OTO = OneToOneLinear(n_input)

        self.encoder = nn.Sequential(nn.Linear(n_input, n_hidden),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(n_hidden),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                )

        self.fc1 = nn.Linear(n_hidden, nz)
        self.fc2 = nn.Linear(n_hidden, nz)

        self.decoder = nn.Sequential(nn.Linear(nz, n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_input),
                                    )

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        # print(mu.device, logvar.device)
        return res, z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        return self.fc1(h), self.fc2(h)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if next(self.parameters()).device == torch.device('cpu'):
            eps = torch.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_().to(next(self.parameters()).device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        return self.decoder(z)

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z
 
    def generate(self, z):
        res = self.decode(z)
        return res


class FC_Classifier(nn.Module):
    """Latent space discriminator"""
    def __init__(self, nz, n_hidden1=1024, n_hidden2=2, n_out=1):
        super(FC_Classifier, self).__init__()
        self.nz = nz
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_out = n_out
        self.decoder = nn.Sequential(
            nn.Linear(nz, n_hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden1, n_hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden1, n_hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden1, n_hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden1, n_hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden1, n_hidden2)
        )
        self.reduction = nn.Sequential(
            nn.Linear(n_hidden2, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.decoder(x)
        out = self.reduction(encoded)
        return torch.squeeze(out), encoded