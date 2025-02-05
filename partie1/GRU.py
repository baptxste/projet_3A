import torch
import torch.nn as nn



############################### modèle gauche droite ############################
class GRUEncoderGD(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, sequence):
        batch_size = sequence.size(0)
        hidden = self.init_hidden(batch_size)  # initialisation des états cachés
        embedded = self.embedding(sequence)  # (batch_size, seq_len, emb_size)
        output, hidden = self.rnn(embedded, hidden)  # hidden contient le dernier état caché et output la suite de tous les états cachés
        return hidden

class GRUDecoderGD(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden, output_len):
        """
        hidden : Dernier état caché de l'encodeur (1, batch_size, hidden_dim).
        """
        batch_size = hidden.size(1)
        outputs = []
        input_t = torch.zeros(batch_size, 1, self.hidden_size)  # entrée initiale
        for _ in range(output_len):
            output, hidden = self.rnn(input_t, hidden)
            output = self.out(output)  # (batch_size, 1, output_size)
            output = self.softmax(output)
            outputs.append(output.squeeze(1))  # (batch_size, output_size)
        return torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)


############################# droite gauche ###############################
class GRUEncoderDG(nn.Module):
    # modèle droite gauche 
    def __init__(self, input_size, emb_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, sequence):
        batch_size = sequence.size(0)
        hidden = self.init_hidden(batch_size)  # initialisation des états cachés
        embedded = self.embedding(sequence)  # (batch_size, seq_len, emb_size)
        
        # Inversion de la séquence
        embedded = torch.flip(embedded, dims=[1])  # (batch_size, seq_len, emb_size)

        output, hidden = self.rnn(embedded, hidden)  
        return hidden

class GRUDecoderDG(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden, output_len):
        """
        hidden : Dernier état caché de l'encodeur (1, batch_size, hidden_dim).
        """
        batch_size = hidden.size(1)
        outputs = []
        input_t = torch.zeros(batch_size, 1, self.hidden_size)  
        for _ in range(output_len):
            output, hidden = self.rnn(input_t, hidden)
            output = self.out(output)  # (batch_size, 1, output_size)
            output = self.softmax(output)
            outputs.append(output.squeeze(1))  # (batch_size, output_size)
        return torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)




############################################## bi dir ####################################
    
class BiGRUEncoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True, bidirectional=True)

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size)  # 2 car bidirectionnel

    def forward(self, sequence):
        batch_size = sequence.size(0)
        hidden = self.init_hidden(batch_size)
        embedded = self.embedding(sequence)

        output, hidden = self.rnn(embedded, hidden)
        
        # on rassemble les états caché des deux sens
        hidden = torch.cat((hidden[0], hidden[1]), dim=-1)  # (batch_size, 2*hidden_size)
        
        return hidden


class BiGRUDecode(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden, output_len):
        batch_size = hidden.size(0)
        outputs = []
        input_t = torch.zeros(batch_size, 1, self.hidden_size)  # entrée initiale
        
        for _ in range(output_len):
            output, hidden = self.rnn(input_t, hidden.unsqueeze(0))  # ajout de la dim couche
            output = self.out(output)
            output = self.softmax(output)
            outputs.append(output.squeeze(1))  

        return torch.stack(outputs, dim=1)
