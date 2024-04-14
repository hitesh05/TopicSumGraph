import torch 
import torch.nn as nn
import torch.nn.functional as F
class NTM_Module(nn.Module):
    
    def __init__(self, dt, K, V): 
        super(NTM_Module, self).__init__()
        self.dt = dt
        self.K = K
        self.V = V
        self.mu = nn.Linear(V, K//2, bias=False)
        self.log_sigma = nn.Linear(V, K//2, bias=False)
        self.W = nn.Linear(K, V, bias=False)
        self.f_psi = nn.Linear(V, dt, bias=False)
        
    def forward(self, G, sentnodes):
        x_bow = G.ndata['bow'][sentnodes]
        mu = F.relu(self.mu(x_bow))
        log_sigma = F.relu(self.log_sigma(x_bow))
        z = torch.cat((mu, log_sigma), -1)
        theta = F.softmax(z, dim=-1)
        x_bow2 = self.W(theta)
        return x_bow2, theta
    

    def get_ti(self):
        # print(self.W.weight.data.shape)
        W_matrix = torch.transpose(self.W.weight.data, 0, 1)
        psi = F.relu(self.f_psi(W_matrix))
        # print(psi.shape)
        return psi # [k, dt]
    
    
