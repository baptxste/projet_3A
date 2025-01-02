import torch
import torch.nn as nn
import torch.nn.functional as F

# class RNNEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers=1):
#         """
#         causal GRU-based Encoder.
#         input_dim : Size input features
#         hidden_dim : Size hidden state
#         num_layers : Number GRU layers
#         """
#         super(RNNEncoder, self).__init__()

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers #nombre de GRU empilées

#         self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

#     def forward(self, x):
#         """
#             x : (batch_size, seq_len, input_dim)
#             outputs : GRU outputs (batch_size, seq_len, hidden_dim) #concaténation des états cachés de la dernière couche
#             hidden : Final hidden state (num_layers, batch_size, hidden_dim) #derniers états cachés concaténés de chaque couche (num_layers * derniers états cachés)
#         """
#         outputs, hidden = self.rnn(x) 
#         return hidden

# class RNNDecoder(nn.Module): 
#     def __init__(self, input_dim, hidden_dim, num_layers=1):
#         super(RNNDecoder, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, input_dim)  # Map hidden state back to input_dim

#     def forward(self, hidden): #on prend approche non-ittérative : estime que pas grande dépendence next token aux previous token sequence
#         """
#             encoder_outputs: Outputs encoder (batch_size, seq_len, hidden_dim)
#             hidden: Final hidden state encoder (num_layers, batch_size, hidden_dim)

#             Reconstructed sequence (batch_size, seq_len, input_dim)
#         """
#         input_t = torch.zeros(hidden.size(1), self.input_dim, self.hidden_dim)
#         outputs, _ = self.rnn(input_t, hidden)  # Pass encoder outputs through decoder GRU
#         outputs = self.fc(outputs)  # Map back to original input dimension
#         return outputs



class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size  
        self.embedding = nn.Embedding(input_size, emb_size)  
        self.rnn = nn.GRU(emb_size, hidden_size) 

    def forward(self, sequence):

        hidden = self.init_hidden(len(sequence))
        sequence = torch.LongTensor([sequence]).transpose(0, 2)  
        for element in sequence:
            embedded = self.embedding(element).transpose(0, 1)
            output, hidden = self.rnn(embedded, hidden)

        return hidden

class RNNDecoder(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(emb_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden, output_len):
        """
            hidden : Dernier état caché encodeur (num_layers, batch_size, hidden_dim).
        """
        outputs = []
        hidden = F.relu(hidden)  

        for _ in range(output_len):
            input_t = torch.zeros(1, hidden.size(1), self.hidden_size)
            output, hidden = self.rnn(input_t, hidden)
            output = self.softmax(self.out(output[0]))
            outputs.append(output)

        return outputs
    
class TensorProductEncoder(nn.Module):
    def __init__(self, n_roles, n_fillers, filler_dim, role_dim, final_layer_width):

        super(TensorProductEncoder, self).__init__()
        
        self.n_roles = n_roles 
        self.n_fillers = n_fillers 
        self.filler_dim = filler_dim #dimension filler embeddings
        self.role_dim = role_dim
        self.final_layer_width = final_layer_width

        self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim) #embedding layer filler
        self.role_embedding = nn.Embedding(self.n_roles, self.role_dim)

        self.last_layer = nn.Linear(self.filler_dim * self.role_dim, self.final_layer_width)
     
    def forward(self, filler_list, role_list):

        fillers_embedded = self.filler_embedding(filler_list)
        roles_embedded = self.role_embedding(role_list)
        
        # sum of flattened tensor products of filler and role embeddings
        output = self.sum_layer(fillers_embedded, roles_embedded)
        
        
        output = self.last_layer(output)
            
        return output


    

# Pipeline
if __name__ == "__main__":
    # Configuration
    batch_size = 1
    seq_len = 5
    input_dim = 3
    hidden_dim = 32
    num_layers = 2

    x = torch.randn(batch_size, seq_len, input_dim)

    encoder = RNNEncoder(input_dim, hidden_dim, num_layers)
    decoder = RNNDecoder(input_dim, hidden_dim, num_layers)

    hidden = encoder(x)

    reconstructed_x = decoder(hidden)

    # Print shapes
    print("Input shape:", x.shape, x)
    print("Reconstructed shape:", reconstructed_x.shape, reconstructed_x)