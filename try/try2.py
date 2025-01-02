# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, emb_size, hidden_size):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(input_size, emb_size)
#         self.rnn = nn.GRU(emb_size, hidden_size)

#     def forward(self, sequence):
#         """
#         Args:
#             sequence: Séquence d'entrée (list ou Tensor) de taille (seq_len,).

#         Returns:
#             hidden: Dernier état caché (1, 1, hidden_size).
#         """
#         embedded = self.embedding(torch.LongTensor(sequence).unsqueeze(1))  # (seq_len, 1, emb_size)
#         _, hidden = self.rnn(embedded)  # Retourne uniquement hidden
#         return hidden


# class RNNDecoder(nn.Module):
#     def __init__(self, emb_size, hidden_size, output_size):
#         super(RNNDecoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.rnn = nn.GRU(emb_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=-1)

#     def forward(self, hidden, output_len, start_token):
#         """
#         Args:
#             hidden: Dernier état caché de l'encodeur (1, 1, hidden_size).
#             output_len: Longueur de la séquence à générer.
#             start_token: Premier token (de taille emb_size).

#         Returns:
#             outputs: Liste des tokens générés (output_len, output_size).
#         """
#         outputs = []
#         input_t = start_token.unsqueeze(0).unsqueeze(0)  # (1, 1, emb_size)

#         for _ in range(output_len):
#             # Corriger la forme de input_t pour être de forme (1, batch_size=1, emb_size)
#             output, hidden = self.rnn(input_t, hidden)
#             output = self.softmax(self.out(output.squeeze(0)))  # (1, output_size)
#             outputs.append(output)
#             input_t = output.unsqueeze(0)  # Ajuste pour être (1, 1, emb_size)

#         return torch.stack(outputs)  # (output_len, 1, output_size)


# # Configuration
# input_size = 10  # Taille du vocabulaire
# emb_size = 8     # Dimension des embeddings
# hidden_size = 16  # Dimension de l'état caché
# seq_len = 5       # Longueur de la séquence
# output_size = input_size  # Taille du vocabulaire (reconstruction)

# # Initialisation des modèles
# encoder = EncoderRNN(input_size, emb_size, hidden_size)
# decoder = RNNDecoder(emb_size, hidden_size, output_size)

# # Données factices
# sequence = [1, 3, 4, 2, 0]  # Séquence d'entrée (exemple)
# output_len = len(sequence)
# start_token = torch.zeros(emb_size)  # Premier token pour le décodeur (par exemple, vecteur nul)

# # Encodeur
# hidden = encoder(sequence)

# # Décodeur
# outputs = decoder(hidden, output_len, start_token)

# # Résultats
# print("Séquence d'entrée :", sequence)
# print("Séquence générée :", torch.argmax(outputs, dim=-1).squeeze().tolist())
