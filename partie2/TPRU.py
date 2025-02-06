import torch 
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class TPRUEncoder(nn.Module):
    def __init__(self, d, d_prime, N): 
        super().__init__()
        self.d, self.d_prime, self.N = d, d_prime, N

        self.Wu = nn.Parameter(torch.randn(d, d))
        self.Wr = nn.Parameter(torch.randn(d, d))
        self.Wb = nn.Parameter(torch.randn(d, d))
        self.W = nn.Parameter(torch.randn(d, d_prime))
        self.Wx = nn.Parameter(torch.randn(d, d_prime))

        self.register_buffer("V", torch.randn(d, N))  # V fixe

    def forward(self, sequence):
        """
        Args:
            sequence: (batch_size, seq_len, d_prime)
        
        Returns:
            bt: (batch_size, N) -> état caché final de chaque séquence du batch
        """
        batch_size, seq_len = sequence.shape 

        sequence = sequence.float() 

        # Initialisation de bt (batch_size, N)
        bt = torch.zeros(batch_size, self.d, device=sequence.device)

        for t in range(seq_len):  
            xt = sequence[:, t].unsqueeze(1)  # (batch_size, d_prime) -> numéro t de chaque séq du batch
            #print("xt:", xt.shape)
            U = self.Wu @ self.V  # (d, N)
            #print("U:", U.shape)

            fxt = F.relu(U.T @ self.W @ xt.T)  # (N, batch_size): xt.T au lieu de xt pour avoir  W@xt : (d,batch)
            fbt = F.relu(U.T @ bt.T)  # (N, batch_size)

            numerator = (fbt + fxt) ** 2
            denominator = torch.sum(numerator, dim=0, keepdim=True) + 1e-6
            ft = numerator / denominator  # (N, batch_size)

            R = self.Wr @ self.V
            b = R @ ft  # (N, batch_size)

            gt = torch.sigmoid(self.Wb @ bt.T + self.Wx @ xt.T)  # (N, batch_size)

            bt = (gt * torch.tanh(b) + (1 - gt) * bt.T).T  # (batch_size, N)
        #print(bt.shape)
        return bt  # (batch_size, N)

# class TPRUEncoder(nn.Module):
#     def __init__(self, d, d_prime, N): 
#         """ 
#         d : hidden size
#         d_prime : 1 pour les chiffres
#         N : nombres de fillers, roles
#         """
#         super().__init__()
#         self.d, self.d_prime, self.N = d, d_prime, N

#         self.Wu = nn.Parameter(torch.randn(d, d))
#         self.Wr = nn.Parameter(torch.randn(d, d))
#         self.Wb = nn.Parameter(torch.randn(d, d))
#         self.W = nn.Parameter(torch.randn(d, d_prime))
#         self.Wx = nn.Parameter(torch.randn(d, d_prime))

#         self.register_buffer("V", torch.randn(d, N)) #V fixe

#     def forward(self, sequence):
#         if sequence.dim() == 1:  
#             sequence = sequence.unsqueeze(1).float() #(len(seq),) -> (len(seq),1)
#         else:
#             sequence = sequence.float()     
#         bt = torch.zeros(self.d, 1)  
#         for xt in sequence:  
#             #print("xt:", xt)
#             xt = xt.view(1, 1)
#             #print("xt:", xt.shape)
#             U = self.Wu @ self.V #dxN
#             #print("U:", U.shape)
#             fxt = F.relu(U.T @ self.W @ xt) 
#             #print("fxt:", fxt.shape)
#             fbt = F.relu(U.T @ bt) 
#             #print("fbt:", fbt.shape) 
#             numerator = (fbt + fxt) ** 2
#             denominator = torch.sum(numerator)
#             ft = numerator / denominator  
#             R = self.Wr @ self.V 
#             b = R @ ft  
#             gt = torch.sigmoid(self.Wb @ bt + self.Wx @ xt) 
#             bt = gt * torch.tanh(b) + (1 - gt) * bt  #dx1
#         return bt



# d = 32 #hidden size
# d_prime = 1 #chiffre
# N = 5 #5 fillers par chiffre de la séquence
# encoder = TPRUEncoder(d, d_prime, N)
# input = torch.tensor([1,3,4])
# print(encoder(input).shape)



# N = 5  
# fxt = torch.tensor([1,0,2])  # Nx1
# fbt = torch.tensor([1,2,1])  # Nx1
# numerator = (fbt + fxt)**2  # Nx1

# denominator = torch.sum(numerator)  # Scalaire

# # Diviser chaque composante par la somme pour obtenir ft
# ft = (numerator / denominator)  # Nx1

# print(numerator)
# print(denominator)
# print(ft)
