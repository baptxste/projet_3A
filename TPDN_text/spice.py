import torch
from torch.utils.data import Dataset
import numpy as np
import re
import pickle
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from torch.nn.utils.rnn import pad_sequence


# class EmbeddingDataset(Dataset):
#     def __init__(self, sentence_tensor):
#         self.sentence_tensor = sentence_tensor

#     def __len__(self):
#         return len(self.sentence_tensor)

#     def __getitem__(self, idx):
#         return self.sentence_tensor[idx]

# class SpiceEmbeddingModel:
#     def __init__(self, emb_dim=50, window_size=3):
#         self.emb_dim = emb_dim
#         self.window_size = window_size
#         self.word2idx = {}
#         self.idx2word = {}
#         self.embeddings = None

#     def preprocess_text(self, text):
#         sentences = text.split('\n')
#         sentences = [lst for lst in sentences if ":" not in lst]
#         text = '\n'.join(sentences).lower()
#         text = re.sub(r"[^a-z\s]", "", text)
#         sentences = [lst for lst in text.split('\n') if lst]
#         return sentences

#     def build_cooccurrence_matrix(self, words):
#         if not self.word2idx:
#             vocab = sorted(set(words))
#             self.word2idx = {word: idx for idx, word in enumerate(vocab)}
#             self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
#         vocab_size = len(self.word2idx)
#         co_matrix = np.zeros((vocab_size, vocab_size))
        
#         for i, word in enumerate(words):
#             word_idx = self.word2idx[word]
#             start, end = max(0, i - self.window_size), min(len(words), i + self.window_size + 1)
#             for j in range(start, end):
#                 if i != j:
#                     neighbor_idx = self.word2idx[words[j]]
#                     co_matrix[word_idx, neighbor_idx] += 1
        
#         return co_matrix

#     def spice_embedding(self, co_matrix):
#         u, s, vt = svds(co_matrix, k=self.emb_dim)
#         embeddings = normalize(u @ np.diag(np.sqrt(s)))
#         return embeddings

#     def save_model(self):
#         with open("spice_embeddings.pkl", "wb") as f:
#             pickle.dump(self.embeddings, f)
#         with open("spice_vocab.pkl", "wb") as f:
#             pickle.dump(self.word2idx, f)

#     def load_model(self):
#         with open("spice_embeddings.pkl", "rb") as f:
#             self.embeddings = pickle.load(f)
#         with open("spice_vocab.pkl", "rb") as f:
#             self.word2idx = pickle.load(f)
#         self.idx2word = {idx: word for word, idx in self.word2idx.items()}

#     def encode_word(self, word):
#         return self.embeddings[self.word2idx[word]] if word in self.word2idx else None

#     def decode_embedding(self, vector, top_n=5):
#         similarities = np.dot(self.embeddings, vector)
#         closest_indices = np.argsort(similarities)[-top_n:][::-1]
#         return [self.idx2word[idx] for idx in closest_indices]

#     def get_dataset(self, text):
#         sentences = self.preprocess_text(text)
#         words = [word for sentence in sentences for word in sentence.split()]
        
#         co_matrix = self.build_cooccurrence_matrix(words)
#         self.embeddings = self.spice_embedding(co_matrix)
#         self.save_model()  

#         sentence_embeddings = [
#             torch.tensor([self.embeddings[self.word2idx[word]] for word in sentence.split() if word in self.word2idx], dtype=torch.float32)
#             for sentence in sentences if sentence.split()
#         ]

#         padded_sentence_tensor = pad_sequence(sentence_embeddings, batch_first=True, padding_value=0.0)
#         return EmbeddingDataset(padded_sentence_tensor)





class EmbeddingAndRoleDataset(Dataset):
    """
    Dataset qui renvoie pour chaque phrase un tuple (fillers, roles)
    - fillers : tenseur (seq_len, emb_dim) des embeddings de mots
    - roles   : tenseur (seq_len, role_dim) des vecteurs de rôle calculés par cosinus
    """
    def __init__(self, fillers_list, roles_list):
        self.fillers_list = fillers_list
        self.roles_list = roles_list

    def __len__(self):
        return len(self.fillers_list)

    def __getitem__(self, idx):
        return self.fillers_list[idx], self.roles_list[idx]

def positional_encoding(seq_len, role_dim, p):
    """
    encoding positionnel pour les rôles un peu comme dans les transformers
    """
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
    dims = torch.arange(role_dim, dtype=torch.float32).unsqueeze(0)         # (1, role_dim)
    pos_enc = torch.cos(2 * positions * dims / p)  # (seq_len, role_dim)
    return pos_enc

def collate_fn_fillers_roles(batch):
    """
    Collate function pour un batch de tuples (fillers, roles).
    Effectue le padding sur les fillers et les roles et renvoie aussi les longueurs.
    """
    fillers_list, roles_list = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in fillers_list], dtype=torch.long)
    padded_fillers = pad_sequence(fillers_list, batch_first=True, padding_value=0.0)
    padded_roles = pad_sequence(roles_list, batch_first=True, padding_value=0.0)
    return padded_fillers, padded_roles, lengths

class SpiceEmbeddingModel:
    def __init__(self, emb_dim=50, window_size=3):
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None

    def preprocess_text(self, text):
        # On retire les lignes contenant ":" puis on met en minuscule et on garde uniquement les lettres et espaces
        sentences = text.split('\n')
        sentences = [lst for lst in sentences if ":" not in lst]
        text = '\n'.join(sentences).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        sentences = [lst for lst in text.split('\n') if lst]
        return sentences

    def build_cooccurrence_matrix(self, words):
        # Si le vocabulaire est vide, on le construit à partir de tous les mots du texte
        if not self.word2idx:
            vocab = sorted(set(words))
            self.word2idx = {word: idx for idx, word in enumerate(vocab)}
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        vocab_size = len(self.word2idx)
        co_matrix = np.zeros((vocab_size, vocab_size))
        for i, word in enumerate(words):
            # Si le mot n'est pas dans le vocabulaire (cas d'un modèle pré-chargé avec un vocabulaire différent),
            # on passe au mot suivant
            if word not in self.word2idx:
                continue
            word_idx = self.word2idx[word]
            start, end = max(0, i - self.window_size), min(len(words), i + self.window_size + 1)
            for j in range(start, end):
                # On vérifie également que le mot voisin est dans le vocabulaire
                if i != j and words[j] in self.word2idx:
                    neighbor_idx = self.word2idx[words[j]]
                    co_matrix[word_idx, neighbor_idx] += 1
        return co_matrix

    def spice_embedding(self, co_matrix):
        # On utilise une décomposition SVD pour obtenir des embeddings de dimension emb_dim
        u, s, vt = svds(co_matrix, k=self.emb_dim)
        embeddings = normalize(u @ np.diag(np.sqrt(s)))
        return embeddings

    def save_model(self):
        with open("spice_embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)
        with open("spice_vocab.pkl", "wb") as f:
            pickle.dump(self.word2idx, f)

    def load_model(self):
        with open("spice_embeddings.pkl", "rb") as f:
            self.embeddings = pickle.load(f)
        with open("spice_vocab.pkl", "rb") as f:
            self.word2idx = pickle.load(f)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode_word(self, word):
        return self.embeddings[self.word2idx[word]] if word in self.word2idx else None

    def decode_embedding(self, vector, top_n=5):
        similarities = np.dot(self.embeddings, vector)
        closest_indices = np.argsort(similarities)[-top_n:][::-1]
        return [self.idx2word[idx] for idx in closest_indices]

    def get_dataset(self, text, role_dim=20):
        """
        Pour un texte en entrée, on :
         - pré-traite pour obtenir des phrases
         - crée les embeddings (fillers) via SVD sur la matrice de cooccurrence
         - pour chaque phrase, on récupère les fillers et on calcule les rôles
           (les rôles sont calculés avec la fonction cosinus en fonction de la position)
         - on renvoie un EmbeddingAndRoleDataset
        """
        sentences = self.preprocess_text(text)
        words = [word for sentence in sentences for word in sentence.split()]
        co_matrix = self.build_cooccurrence_matrix(words)
        self.embeddings = self.spice_embedding(co_matrix)
        self.save_model()

        # Déterminer la longueur maximale (pour fixer l'échelle p)
        max_seq_len = max(len(sentence.split()) for sentence in sentences)

        fillers_list = []
        roles_list = []
        for sentence in sentences:
            tokens = sentence.split()
            if not tokens:
                continue
            # Récupérer les fillers (embeddings) pour chaque mot de la phrase
            sentence_fillers = []
            for word in tokens:
                if word in self.word2idx:
                    emb = self.embeddings[self.word2idx[word]]
                    sentence_fillers.append(torch.tensor(emb, dtype=torch.float32))
            if len(sentence_fillers) == 0:
                continue
            sentence_fillers = torch.stack(sentence_fillers)  # (seq_len, emb_dim)
            fillers_list.append(sentence_fillers)
            # Calculer les rôles via un positional encoding simple
            seq_len = sentence_fillers.size(0)
            sentence_roles = positional_encoding(seq_len, role_dim, p=max_seq_len)
            roles_list.append(sentence_roles)

        return EmbeddingAndRoleDataset(fillers_list, roles_list)

