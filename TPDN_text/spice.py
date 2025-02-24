import torch
from torch.utils.data import Dataset
import numpy as np
import re
import pickle
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from torch.nn.utils.rnn import pad_sequence


class EmbeddingDataset(Dataset):
    def __init__(self, sentence_tensor):
        self.sentence_tensor = sentence_tensor

    def __len__(self):
        return len(self.sentence_tensor)

    def __getitem__(self, idx):
        return self.sentence_tensor[idx]

class SpiceEmbeddingModel:
    def __init__(self, emb_dim=50, window_size=3):
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None

    def preprocess_text(self, text):
        sentences = text.split('\n')
        sentences = [lst for lst in sentences if ":" not in lst]
        text = '\n'.join(sentences).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        sentences = [lst for lst in text.split('\n') if lst]
        return sentences

    def build_cooccurrence_matrix(self, words):
        if not self.word2idx:
            vocab = sorted(set(words))
            self.word2idx = {word: idx for idx, word in enumerate(vocab)}
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        vocab_size = len(self.word2idx)
        co_matrix = np.zeros((vocab_size, vocab_size))
        
        for i, word in enumerate(words):
            word_idx = self.word2idx[word]
            start, end = max(0, i - self.window_size), min(len(words), i + self.window_size + 1)
            for j in range(start, end):
                if i != j:
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

    def get_dataset(self, text):
        sentences = self.preprocess_text(text)
        words = [word for sentence in sentences for word in sentence.split()]
        
        co_matrix = self.build_cooccurrence_matrix(words)
        self.embeddings = self.spice_embedding(co_matrix)
        self.save_model()  

        sentence_embeddings = [
            torch.tensor([self.embeddings[self.word2idx[word]] for word in sentence.split() if word in self.word2idx], dtype=torch.float32)
            for sentence in sentences if sentence.split()
        ]

        padded_sentence_tensor = pad_sequence(sentence_embeddings, batch_first=True, padding_value=0.0)
        return EmbeddingDataset(padded_sentence_tensor)