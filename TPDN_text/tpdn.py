import torch
import torch.nn as nn


class TPDNEncoder(nn.Module):
    def __init__(self, filler_dim, role_dim, output_dim):
        """
        Args:
            filler_dim (int): Dimension des embeddings de mots.
            role_dim (int): Dimension des vecteurs de rôle.
            output_dim (int): Dimension de sortie (représentation latente).
        """
        super(TPDNEncoder, self).__init__()
        self.linear = nn.Linear(filler_dim * role_dim, output_dim)
    
    def forward(self, fillers, roles, lengths):
        """
        Args:
            fillers: Tensor de forme (batch, seq_len, filler_dim)
            roles  : Tensor de forme (batch, seq_len, role_dim)
            lengths: Tensor de forme (batch,) indiquant la longueur réelle de chaque séquence
        Retourne:
            output: Tensor de forme (batch, output_dim)
        """
        batch, seq_len, _ = fillers.size()
        # Calcul du produit tensoriel (outer product) pour chaque token
        tensor_products = torch.einsum('bse,bsr->bser', fillers, roles)  # (batch, seq_len, filler_dim, role_dim)
        device = fillers.device
        # Création d'un masque pour ignorer le padding
        mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).unsqueeze(-1).float()  # (batch, seq_len, 1, 1)
        tensor_products = tensor_products * mask
        # Somme sur la dimension de la séquence
        tensor_sum = tensor_products.sum(dim=1)  # (batch, filler_dim, role_dim)
        # Aplatissement
        tensor_flat = tensor_sum.view(batch, -1)   # (batch, filler_dim * role_dim)
        output = self.linear(tensor_flat)
        return output