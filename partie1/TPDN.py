import torch
import torch.nn as nn


class TensorProductEncoder(nn.Module):
    def __init__(self, n_roles, n_fillers, filler_dim, role_dim, hidden_size):
        super().__init__()

        self.n_roles = n_roles
        self.n_fillers = n_fillers
        self.filler_dim = filler_dim  # dimension des embeddings des fillers
        self.role_dim = role_dim  # dimension des embeddings des rôles

        self.hidden_size = hidden_size  # taille du vecteur caché final

        # embedding pour les fillers et les rôles
        self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim)
        self.role_embedding = nn.Embedding(self.n_roles, self.role_dim)

        # couche linéaire pour adapter la sortie du produit tensoriel à la taille cachée attendue
        self.last_layer = nn.Linear(self.filler_dim * self.role_dim, self.hidden_size)

    def forward(self, filler_list, role_list):
        fillers_embedded = self.filler_embedding(filler_list)  # (batch_size, seq_len, filler_dim)
        roles_embedded = self.role_embedding(role_list)  # (batch_size, seq_len, role_dim)

        #  stocker les différentes sorties 
        batch_size = fillers_embedded.size(0)
        seq_len = fillers_embedded.size(1)
        final_states = torch.zeros(batch_size, seq_len, self.hidden_size)



        for i in range(fillers_embedded.size(1)):  # itérer sur chaque élément de la séquence
 
            fillers_expanded = fillers_embedded[:, i].unsqueeze(-1)  # (batch_size, filler_dim, 1)
            roles_expanded = roles_embedded[:, i].unsqueeze(-2)  # (batch_size, 1, role_dim)
            
            tensor_product = torch.matmul(fillers_expanded, roles_expanded)  # (batch_size, filler_dim, role_dim)
            

            tensor_product_flattened = tensor_product.view(tensor_product.size(0), -1)  # (batch_size, filler_dim * role_dim)
            

            final_state = self.last_layer(tensor_product_flattened)  # (batch_size, hidden_size)
            
     
            final_states[:, i, :] = final_state

        # final_states est maintenant une liste des états finaux pour chaque étape
        # Le dernier élément est l'état final correspondant à la séquence complète
        return final_states
    


# Ancien bug 
"""
class TensorProductEncoder(nn.Module):
    def __init__(self, n_roles, n_fillers, filler_dim, role_dim, hidden_size):
        super().__init__()
        
        self.n_roles = n_roles
        self.n_fillers = n_fillers
        self.filler_dim = filler_dim  # dimension filler embeddings
        self.role_dim = role_dim
        self.hidden_size = hidden_size 

        self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim)
        self.role_embedding = nn.Embedding(self.n_roles, self.role_dim)

        # couche linéaire pour adapter la sortie au `hidden_size` du GRUEncoder
        self.last_layer = nn.Linear(self.filler_dim * self.role_dim, self.hidden_size)

    def forward(self, filler_list, role_list):
        fillers_embedded = self.filler_embedding(filler_list)  # (batch_size, seq_len, filler_dim)
        roles_embedded = self.role_embedding(role_list)        # (batch_size, seq_len, role_dim)

        # produit tensoriel
        fillers_expanded = fillers_embedded.unsqueeze(-1)  # (batch_size, seq_len, filler_dim, 1)
        roles_expanded = roles_embedded.unsqueeze(-2)      # (batch_size, seq_len, 1, role_dim)
        tensor_product = torch.matmul(fillers_expanded, roles_expanded)  # (batch_size, seq_len, filler_dim, role_dim)

        tensor_product_flattened = tensor_product.view(tensor_product.size(0), tensor_product.size(1), -1)
        summed_output = tensor_product_flattened.sum(dim=1)  # (batch_size, filler_dim * role_dim)

        output = self.last_layer(summed_output)  # (batch_size, hidden_size) A ENLEVER

        return output
"""