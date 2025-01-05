import torch
import torch.nn as nn


# class TensorProductEncoder(nn.Module):
#     def __init__(self, n_roles, n_fillers, filler_dim, role_dim, final_layer_width):

#         super(TensorProductEncoder, self).__init__()
        
#         self.n_roles = n_roles 
#         self.n_fillers = n_fillers 
#         self.filler_dim = filler_dim #dimension filler embeddings
#         self.role_dim = role_dim
#         self.final_layer_width = final_layer_width

#         self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim) #embedding layer filler
#         self.role_embedding = nn.Embedding(self.n_roles, self.role_dim)

#         self.last_layer = nn.Linear(self.filler_dim * self.role_dim, self.final_layer_width)
     
#     def forward(self, filler_list, role_list):

#         fillers_embedded = self.filler_embedding(filler_list)
#         roles_embedded = self.role_embedding(role_list)
        
#         # sum of flattened tensor products of filler and role embeddings
#         output = self.sum_layer(fillers_embedded, roles_embedded)
        
        
#         output = self.last_layer(output)
            
#         return output


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

        # Couche linéaire pour adapter la sortie au `hidden_size` du GRUEncoder
        self.last_layer = nn.Linear(self.filler_dim * self.role_dim, self.hidden_size)

    def forward(self, filler_list, role_list):
        fillers_embedded = self.filler_embedding(filler_list)  # (batch_size, seq_len, filler_dim)
        roles_embedded = self.role_embedding(role_list)        # (batch_size, seq_len, role_dim)

        # Produit tensoriel
        fillers_expanded = fillers_embedded.unsqueeze(-1)  # (batch_size, seq_len, filler_dim, 1)
        roles_expanded = roles_embedded.unsqueeze(-2)      # (batch_size, seq_len, 1, role_dim)
        tensor_product = torch.matmul(fillers_expanded, roles_expanded)  # (batch_size, seq_len, filler_dim, role_dim)

        # Aplatissement et somme sur seq_len
        tensor_product_flattened = tensor_product.view(tensor_product.size(0), tensor_product.size(1), -1)
        summed_output = tensor_product_flattened.sum(dim=1)  # (batch_size, filler_dim * role_dim)

        # Couche linéaire pour obtenir la taille `hidden_size`
        output = self.last_layer(summed_output)  # (batch_size, hidden_size)

        return output
