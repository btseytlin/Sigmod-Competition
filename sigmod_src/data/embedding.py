import os
import dill
import gensim
import nltk
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field
from sklearn.preprocessing import normalize
from annoy import AnnoyIndex

def tokenize(text):
    return text.split()

def get_fields():
    text_field = Field(sequential=True, tokenize=tokenize, lower=True)
    return text_field

class LSTMEmbedder(nn.Module):
    def __init__(self,
                 text_field=None,
                 hidden_dim=100,
                 emb_dim=100,
                 spatial_dropout=0.05, 
                 recurrent_dropout=0.1, 
                 num_linear=1,
                 num_lstm=1,
                 **kwargs):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.spatial_dropout = spatial_dropout
        self.recurrent_dropout = recurrent_dropout
        self.num_linear = num_linear
        self.num_lstm = num_lstm

        self.text_field = text_field

        self.embedding = nn.Embedding(len(text_field.vocab), emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim,
                               num_layers=num_lstm, 
                               dropout=recurrent_dropout)
        self.linear_layers = []
        for _ in range(num_linear):
            self.linear_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), 
                )
            )
        self.linear_layers = nn.ModuleList(self.linear_layers)

    def configuration_dict(self):
        return dict(hidden_dim=self.hidden_dim,
                    emb_dim=self.emb_dim,
                    spatial_dropout=self.spatial_dropout,
                    recurrent_dropout=self.recurrent_dropout,
                    num_linear=self.num_linear,
                    num_lstm=self.num_lstm)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        return feature
    
    def infer(self, texts):
        tokens = [tokenize(t) for t in texts]
        tensor = self.text_field.numericalize(tokens)
        return self(tensor)


class Embedder:
    def __init__(self, 
                 model=None,
                 ids=None,
                 texts=None,
                 index_trees=300):
        self.model = model
        self.index_trees = index_trees
        self.ids = ids
        self.texts = texts
        self.emb_dim = None
        self.indexer = None

    def get_params_dict(self):
        return dict(
                emb_dim=self.emb_dim,
                index_trees=self.index_trees,
                texts=self.texts,
                ids=self.ids)

    def set_params_dict(self, params):
        for k, v in params.items():
            setattr(self, k, v)
    
    def make_indexer(self, embeddings):
        indexer = AnnoyIndex(len(embeddings[0]), 'angular')
        for i, emb in enumerate(embeddings):
            indexer.add_item(i, emb)

        indexer.build(10000)
        return indexer

    def fit(self, embeddings):
        self.emb_dim = len(embeddings[0])
        self.indexer = self.make_indexer(normalize(embeddings))

    def lookup(self, text, n=10, **kwargs):
        vector = self.model.infer([text])[0]
        neighboor_idx = self.indexer.get_nns_by_vector(normalize(vector), n, **kwargs)
        return neighboor_idx
    
    def lookup_ids(self, text, n=10, **kwargs):
        return self.ids[self.lookup(text, n=n, **kwargs)]
    
    def lookup_texts(self, text, n=10, **kwargs):
        return self.texts[self.lookup(text, n=n, **kwargs)]

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        dill.dump(self.get_params_dict(), open( os.path.join(dir_path, 'emb.params'), "wb" ))
        dill.dump(self.model.configuration_dict(), open(os.path.join(dir_path, 'emb.nnconfig'), "wb" ))
        dill.dump(self.model.text_field, open( os.path.join(dir_path, 'emb.text_field'), "wb" ))
        torch.save(self.model.state_dict(), os.path.join(dir_path, 'emb.pt'))
        self.indexer.save(os.path.join(dir_path, 'emb.annoy'))

    @classmethod
    def load(cls, dir_path):
        params = dill.load(open(os.path.join(dir_path, 'emb.params'), "rb" ))
        emb = Embedder()
        emb.set_params_dict(params)
        
        configuration_dict = dill.load(open(os.path.join(dir_path, 'emb.nnconfig'), "rb" ))
        text_field = dill.load(open(os.path.join(dir_path, 'emb.text_field'), "rb" ))
        state_dict = torch.load(os.path.join(dir_path, 'emb.pt'))

        model = LSTMEmbedder(text_field=text_field, **configuration_dict)
        model.load_state_dict(state_dict)
        model.eval()
        
        indexer = AnnoyIndex(emb.emb_dim, 'angular')
        indexer.load(os.path.join(dir_path, 'emb.annoy'))
        
        emb.model = model
        emb.indexer = indexer
        return emb