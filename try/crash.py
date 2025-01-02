# import torch.nn as nn
# import torch

# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, emb_size, hidden_size):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size # Hidden size
#         self.embedding = nn.Embedding(input_size, emb_size) # Embedding layer
#         self.rnn = nn.GRU(emb_size, hidden_size) # Recurrent layer
     
#     def forward(self, sequence):
#         hidden = torch.zeros(1,len(sequence),self.hidden_size)
#         # sequence : batch*input_size, [sequence] : 1*batch*input_size, transpose : input*batch*1
#         sequence = torch.LongTensor([sequence]).transpose(0,2) 
       

#         for element in sequence:
#             embedded = self.embedding(element).transpose(0,1)
#             output, hidden = self.rnn(embedded, hidden)
            
#         return hidden
    
# if __name__ == '__main__' :
#     sequence = [[1,2,4],[3,4,5]]
#     # sequence = torch.LongTensor([sequence])
#     # # print(sequence.shape)
#     # sequence=sequence.transpose(0,2)
#     # # print(sequence.shape)
#     # for element in sequence :
#     #     print(element)
#     #     E=nn.Embedding(3, 4)
#     #     embedded = E(element).transpose(0,1)
#     #     print(embedded)

#     encoder = EncoderRNN(3, 4, 10)
#     print(encoder(sequence))

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size): # !!!!!!!!! input_size = voc size pas longueur séquence entrée
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.embedding = nn.Embedding(input_size, emb_size)  
        self.rnn = nn.GRU(emb_size, hidden_size)  

    def forward(self, sequence):
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.LongTensor(sequence)

        # Taille initiale de hidden : (num_layers * num_directions, batch_size, hidden_size)
        batch_size = sequence.size(0)  
        hidden = torch.zeros(1, batch_size, self.hidden_size)  # Initialisation du tenseur hidden

        sequence = sequence.transpose(0, 1)  # (seq_len, batch_size)

        # Boucle itérative pour traiter chaque élément de la séquence
        for element in sequence:
            element = element.unsqueeze(0)  # (1, batch_size) !!!! faire unsqueeze
            embedded = self.embedding(element)  # (1, batch_size, emb_size)
            _, hidden = self.rnn(embedded, hidden)  # Met à jour l'état caché

        return hidden

class DecoderRNN(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size, output_len): # !!!! output_size : Nombre total de mots dans le vocabulaire de sortie = input size
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size  
        self.output_size = output_size  
        self.emb_size = emb_size  
        self.rnn = nn.GRU(emb_size, hidden_size) 
        self.out = nn.Linear(hidden_size, output_size)  # Couche linéaire pour produire la sortie
        self.softmax = nn.LogSoftmax(dim=1) 
        self.output_len=output_len # Longueur de la séquence de sortie (même que la séquence d'entrée pour autoencodeur)

    def forward(self, hidden):
        outputs = []

        input_emb = torch.zeros(1, hidden.size(1), self.emb_size)  # (1, batch_size, emb_size)

        for _ in range(self.output_len):
            output, hidden = self.rnn(input_emb, hidden)  
            output = self.softmax(self.out(output[0]))  # Calcul des probabilités de sortie
            outputs.append(output)

            # Prépare l'entrée suivante (dans ce cas, vecteur nul pour simplifier)
            input_emb = torch.zeros(1, hidden.size(1), self.emb_size)  # Peut être remplacé par une vraie entrée
        outputs_tensor = torch.stack(outputs, dim=0)  # (output_len, batch_size, output_size)

        # Change l'ordre des dimensions : (batch_size, output_len, output_size)
        outputs_tensor = outputs_tensor.permute(1, 0, 2)
        # Applique argmax pour obtenir les indices prédits
        predicted_indices = torch.argmax(outputs_tensor, dim=2).float()  # (batch_size, output_len)
        # predicted_indices_list = predicted_indices.tolist()
        return predicted_indices

# class DecoderRNN(nn.Module):
#     def __init__(self, output_size, emb_size, hidden_size, output_len):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size  
#         self.output_size = output_size  
#         self.emb_size = emb_size  
#         self.rnn = nn.GRU(emb_size, hidden_size) 
#         self.out = nn.Linear(hidden_size, output_size)  # Couche linéaire pour produire la sortie
#         self.softmax = nn.LogSoftmax(dim=1) 
#         self.output_len = output_len  # Longueur de la séquence de sortie

#     def forward(self, hidden):
#         outputs = []

#         input_emb = torch.zeros(1, hidden.size(1), self.emb_size)  # (1, batch_size, emb_size)

#         for _ in range(self.output_len):
#             output, hidden = self.rnn(input_emb, hidden)  
#             output = self.out(output[0])  # Pas de softmax ici, on garde les logits
#             outputs.append(output)

#             # Prépare l'entrée suivante (peut être amélioré pour les modèles autoregressifs)
#             input_emb = torch.zeros(1, hidden.size(1), self.emb_size)

#         outputs_tensor = torch.stack(outputs, dim=0)  # (output_len, batch_size, output_size)
#         outputs_tensor = outputs_tensor.permute(1, 0, 2)  # (batch_size, output_len, output_size)
#         return outputs_tensor  # Retourne les logits


# Exemple d'utilisation
input_size = 10  # Nombre total de mots dans le vocabulaire d'entrée
emb_size = 16    # Taille des vecteurs d'embedding
hidden_size = 32 # Taille des vecteurs d'état caché
output_size = 10 # Nombre total de mots dans le vocabulaire de sortie

encoder = EncoderRNN(input_size, emb_size, hidden_size)
decoder = DecoderRNN(output_size, emb_size, hidden_size, output_len=4) 


sequence = [[1, 2, 3, 4],[5,6,7,8]]  # Séquence d'exemple
sequence = torch.LongTensor(sequence)  # !!!!! dimension batch
hidden = encoder(sequence)


outputs = decoder(hidden)
# print(outputs)
# print(outputs)
# predicted_indices = [torch.argmax(output, dim=1) for output in outputs]

# # Si vous voulez les convertir en une liste lisible (par exemple, pour impression)
# predicted_indices_list = [indices.tolist() for indices in predicted_indices]

# print(predicted_indices_list)



# print(len(outputs))


# print(outputs[0].shape)
# decoded_sequence = [torch.argmax(output, dim=1).item() for output in outputs]
# print(decoded_sequence.shape)






















# # Exemple d'utilisation
# input_size = 10  # Nombre total de mots dans le vocabulaire
# emb_size = 16    # Taille des vecteurs d'embedding
# hidden_size = 32 # Taille des vecteurs d'état caché

# encoder = EncoderRNN(input_size, emb_size, hidden_size)
# sequence = [1, 2, 3, 4]  # Séquence d'exemple
# sequence = torch.LongTensor([sequence])  # Ajoute une dimension batch
# hidden = encoder(sequence)
# print("Hidden state shape:", hidden.shape)

# sequence = torch.LongTensor([[1,2,3], [4,5,6]])
# # print(sequence.shape)
# sequence = sequence.transpose(0, 1)  # (seq_len, batch_size)
# # print(sequence)
# E = nn.Embedding(input_size, emb_size)
# for t in range(sequence.size(0)):
#     element = sequence[t].unsqueeze(0)  # (1, batch_size)
#     print(element)
#     # embedded = E(element)  # (1, batch_size, emb_size)
#     # print(embedded.shape)

# for element in sequence :
#     print(element.unsqueeze(0))
#     embedded = E(element).transpose(0,1)
#     # print(embedded)
