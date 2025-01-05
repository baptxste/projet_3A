# import os
# print(os.getcwd())
# print(os.listdir())

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import RNNEncoder, RNNDecoder
# from crash import EncoderRNN, DecoderRNN
import torch.nn.functional as F

batch_size = 4
seq_len = 10
input_dim = 1  # Une seule caractéristique par élément de la séquence (nombres aléatoires)
# hidden_dim = 32
# num_layers = 2
epochs = 100
learning_rate = 0.001

##### float 0-1

# def generate_data(batch_size, seq_len): #float 0-1
#     return torch.rand(batch_size, seq_len, input_dim)

# def precision(predicted, target, tolerance=0.05): #seuil pour 0-1
#     """
#     prédiction correcte si différence absoulue < seuil tolérance
#     """
#     correct = torch.abs(predicted - target) < tolerance  # Tensor de booléens
#     return (correct.float().mean().item()) * 100  # Moyenne des booléens * 100

##### entiers 0-10 

def generate_data(batch_size, seq_len, min_val=0, max_val=10):
    return torch.randint(min_val, max_val, (batch_size, seq_len))

def precision(predicted, target):
    correct = (predicted.round() == target)  # Égalité exacte
    return (correct.float().mean().item()) * 100  # Moyenne booléens * 100

# def precision(predicted, target):
#     """
#     Calcule la précision en comparant les indices prédits avec les cibles.
#     Args:
#         predicted: Tenseur de probabilités ou logits, dimensions (batch_size, seq_len, output_size).
#         target: Tenseur d'indices des cibles, dimensions (batch_size, seq_len).
#     """
#     # Trouve les indices prédits en prenant argmax le long de la dernière dimension
#     predicted_indices = torch.argmax(predicted, dim=2)  # (batch_size, seq_len)

#     # Calcule la précision comme pourcentage de correspondances exactes
#     correct = (predicted_indices == target)  # Tensor booléen
#     return (correct.float().mean().item()) * 100  # Moyenne des booléens * 100


# encoder = RNNEncoder(input_dim, hidden_dim, num_layers)
# decoder = RNNDecoder(input_dim, hidden_dim, num_layers)

input_size = 10  # Nombre total de mots dentrée
emb_size = 16    # Taille des vecteurs d'embedding
hidden_size = 32 # Taille des vecteurs d'état caché
output_size = 10 # Nombre total de mots dans le vocabulaire de sortie

encoder = RNNEncoder(input_size, emb_size, hidden_size)
decoder = RNNDecoder(output_size, emb_size, hidden_size, output_len=seq_len)


criterion = nn.MSELoss()  
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

for epoch in range(epochs):

    x = generate_data(batch_size, seq_len)
    target = x.clone()  

    hidden = encoder(x)

    reconstructed_x = decoder(hidden)
    # target = F.one_hot(target.long(), num_classes=output_size).float()
    print(reconstructed_x)

    loss = criterion(reconstructed_x, target)

    accuracy = precision(reconstructed_x, target)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")


print("Entraînement terminé")
torch.save(encoder.state_dict(), "encoder_weights2.pth")
torch.save(decoder.state_dict(), "decoder_weights2.pth")
print("Poids des modèles sauvegardés.")

# x_test = generate_data(1, seq_len).float()
# hidden = encoder(x_test)
# reconstructed_x = decoder(hidden)
# print("x_test :", x_test)
# print("reconstructed_x :", reconstructed_x)
# print("arrondi :", reconstructed_x.round())


