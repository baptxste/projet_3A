import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=num_layers)

    def forward(self, x, lengths):
        lengths, perm_idx = lengths.sort(descending=True)
        x = x[perm_idx]

        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, hidden = self.rnn(packed_x)  

        return hidden[-1], perm_idx  


class GRUDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super().__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, z, lengths):
        max_seq_len = max(lengths)

        # répéter l'état caché pour toute la séquence
        z = z.unsqueeze(1).repeat(1, max_seq_len, 1)

        output, _ = self.rnn(z)
        output = self.fc(output)

        return output

