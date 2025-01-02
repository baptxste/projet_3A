from model import RNNEncoder, TensorProductEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

batch_size = 4
seq_len = 10
input_dim = 1  
hidden_dim = 32
num_layers = 2
epochs = 600
learning_rate = 0.001

encoder = RNNEncoder(input_dim, hidden_dim, num_layers)
encoder.load_state_dict(torch.load("encoder_weights.pth"))

tprencoder = TensorProductEncoder(n_roles=seq_len, n_fillers=seq_len, filler_dim=hidden_dim, role_dim=hidden_dim, final_layer_width=seq_len)

def generate_data(batch_size, seq_len, min_val=0, max_val=10):
    data = torch.randint(min_val, max_val, (batch_size, seq_len, input_dim))
    indices = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, input_dim)  #rôles
    return data, indices

#print(generate_data(2,5))
#print(generate_data(2,5)[0].shape, generate_data(2,5)[1].shape)

criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(tprencoder.parameters()), lr=learning_rate)

for epoch in range(epochs):
    x = generate_data(batch_size, seq_len)
    