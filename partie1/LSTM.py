import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, batch_first=True)

    def init_hidden(self, batch_size):
        # initialisation des états cachés et de cellule à zéro
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, sequence):
        batch_size = sequence.size(0)
        hidden = self.init_hidden(batch_size) # (hidden_state, cell_state)
        embedded = self.embedding(sequence)  # (batch_size, seq_len, emb_size)
        output, hidden = self.rnn(embedded, hidden) 
        return hidden 

class LSTMDecoder(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden, output_len):
        """
        hidden : tuple (hidden_state, cell_state) de l'encodeur.
        """
        batch_size = hidden[0].size(1)  
        outputs = []
        input_t = torch.zeros(batch_size, 1, self.hidden_size)  
        hidden_state, cell_state = hidden 

        for _ in range(output_len):
            output, (hidden_state, cell_state) = self.rnn(input_t, (hidden_state, cell_state))
            output = self.out(output)  # (batch_size, 1, output_size)
            output = self.softmax(output)
            outputs.append(output.squeeze(1))  # (batch_size, output_size)

        return torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
