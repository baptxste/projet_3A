import torch
from torch.utils.data import Dataset
import numpy as np
import re
import pickle
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from torch.nn.utils.rnn import pad_sequence


def collate_fn_fillers_roles(batch):
    """
    Collate function pour un batch de tuples (fillers, roles).
    padding sur les fillers et les roles et renvoie aussi les longueurs.
    voir pour enlever le padding ici car on l'a rajouter dans le get_dataset sinon ca bug et ca apprend pas 
    """
    seq_fillers_list, seq_roles_list = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in seq_fillers_list], dtype=torch.long)
    padded_fillers = pad_sequence(seq_fillers_list, batch_first=True, padding_value=0.0)
    padded_roles = pad_sequence(seq_roles_list, batch_first=True, padding_value=0.0)
    return padded_fillers, padded_roles, lengths



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
        #e construit le vocabulaire à partir de tous les mots du texte
        if not self.word2idx:
            vocab = sorted(set(words))
            self.word2idx = {word: idx for idx, word in enumerate(vocab)}
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        vocab_size = len(self.word2idx)
        co_matrix = np.zeros((vocab_size, vocab_size))
        for i, word in enumerate(words):
            if word not in self.word2idx:
                continue
            word_idx = self.word2idx[word]
            start, end = max(0, i - self.window_size), min(len(words), i + self.window_size + 1)
            for j in range(start, end):
                # on vérifie également que le mot voisin est dans le vocabulaire
                if i != j and words[j] in self.word2idx:
                    neighbor_idx = self.word2idx[words[j]]
                    co_matrix[word_idx, neighbor_idx] += 1
        return co_matrix

    def spice_embedding(self, co_matrix):
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
        sentences = self.preprocess_text(text)
        words = [word for sentence in sentences for word in sentence.split()]
        
        co_matrix = self.build_cooccurrence_matrix(words)
        self.embeddings = self.spice_embedding(co_matrix)
        self.save_model()  
        #  longueur maximale p pour le positionnal encoding
        max_seq_len = max(len(sentence.split()) for sentence in sentences)

        fillers = [
            torch.tensor([self.embeddings[self.word2idx[word]] for word in sentence.split() if word in self.word2idx], dtype=torch.float32)
            for sentence in sentences if sentence.split()
        ]
        roles = [
            positional_encoding(len(sentence.split()), role_dim, p=max_seq_len)  for sentence in sentences
        ]
        padded_fillers_tensor = pad_sequence(fillers, batch_first=True, padding_value=0.0)

        padded_roles_tensor = pad_sequence(roles, batch_first=True, padding_value=0.0)
        return EmbeddingAndRoleDataset(padded_fillers_tensor, padded_roles_tensor)



