import torch 
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class TPDUEncoder(nn.Module):
    def __init__(self, input_dim, binding_dim, output_dim): 
        ''' 
        input_dim = d' : Dimension des entrées
        binding_dim = N : Nombre de filtres par élément d'entrée
        output_dim = d : Dimension de l'état caché
        '''
        super().__init__()
        self.output_dim = output_dim
        
        self.Wu = nn.Linear(output_dim, output_dim, bias=False)
        self.Wr = nn.Linear(output_dim, output_dim, bias=False)
        self.Wb = nn.Linear(output_dim, output_dim, bias=True)
        self.Wx = nn.Linear(input_dim, output_dim, bias=True)
        self.W = nn.Linear(input_dim, binding_dim, bias=False)

        self.V = torch.randn(input_dim, binding_dim)

    def forward(self, sequence):
        bt = torch.zeros(self.output_dim, 1)  # initialiser bt à 0
        for xt in sequence:  
            U = self.Wu(self.V.T)  # (output_dim, input_dim)
            fxt = F.relu(U.T @ self.W(xt))  # (binding_dim, 1)
            fbt = F.relu(U.T @ bt)  # (binding_dim, 1)
            numerator = (fbt + fxt) ** 2
            denominator = torch.sum(numerator)
            ft = numerator / denominator  # (binding_dim, 1)
            R = self.Wr(self.V.T)  #(output_dim, input _dim)
            b = R @ ft  # (output_dim, 1)
            gt = torch.sigmoid(self.Wb(bt) + self.Wx(xt))  # (output_dim, 1)
            bt = gt * torch.tanh(b) + (1 - gt) * bt  # (output_dim, 1)

        return bt


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

input_dim = 1 #chiffre
binding_dim = 5 #5 fillers par chiffre de la séquence
output_dim = 32
encoder = TPDUEncoder(input_dim, binding_dim, output_dim)
print(encoder([1,3,4]))

