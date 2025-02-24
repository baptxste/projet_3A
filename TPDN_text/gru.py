import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


# class GRUEncoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super().__init__()
#         self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers)

#     def forward(self, x, lengths):
#         # x : [batch_size, seq_len, input_size]
#         lengths, perm_idx = lengths.sort(descending=True) # perm_idx contient l'index des positon de départ, on doit trier pour la fonction pad
#         x = x[perm_idx]

#         packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
#         _, (hidden, _) = self.rnn(packed_x) # hidden : [num_layers, batch_size, hidden_size]

#         return hidden[-1], perm_idx  # [batch_size, hidden_size]


# class GRUDecoder(nn.Module):
#     def __init__(self, hidden_size, output_size, num_layers):
#         super().__init__()
#         self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, z, lengths):
#         max_seq_len = max(lengths)
#         z = z.unsqueeze(1).expand(-1, max_seq_len, -1)

#         output, _ = self.rnn(z)
#         output = self.fc(output)

#         return output


# class RNNAutoencoder(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(RNNAutoencoder, self).__init__()
#         self.encoder = RNNEncoder(input_size, hidden_size)
#         self.decoder = RNNDecoder(hidden_size, input_size)

#     def forward(self, x, lengths):
#         z, perm_idx = self.encoder(x, lengths)
#         reconstructed = self.decoder(z, lengths)
#         return reconstructed, perm_idx




class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=num_layers)

    def forward(self, x, lengths):
        # Trier les séquences par longueur décroissante
        lengths, perm_idx = lengths.sort(descending=True)
        x = x[perm_idx]

        # Emballer les séquences pour éviter de calculer sur le padding
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, hidden = self.rnn(packed_x)  # GRU ne renvoie que hidden_state

        return hidden[-1], perm_idx  # Retourne le dernier état caché [batch_size, hidden_size]


class GRUDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super().__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, z, lengths):
        max_seq_len = max(lengths)

        # Répéter l'état caché pour toute la séquence
        z = z.unsqueeze(1).repeat(1, max_seq_len, 1)

        output, _ = self.rnn(z)
        output = self.fc(output)

        return output
