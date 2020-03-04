import os
import pickle
import gensim
import nltk
from sklearn.model_selection import train_test_split
from gensim.similarities.index import AnnoyIndexer
from gensim.models.doc2vec import Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec


def tokenize_doc(doc):
    return nltk.tokenize.word_tokenize(doc)

def tokenize_docs(docs):
    for doc in docs:
        yield tokenize_doc(doc)

def tag_docs(docs, ids):
    for doc_id, doc in zip(ids, docs):
        tokens = tokenize_doc(doc)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [doc_id])

class Embedder:
    def __init__(self, vector_size=200,
                       train_epochs=300,
                       index_trees=1000):
        self.vector_size = vector_size
        self.train_epochs = train_epochs
        self.index_trees = index_trees
        self.texts = []
        self.ids = []

        self.doc2vec = None
        self.indexer = None

    def get_params_dict(self):
        return dict(vector_size=self.vector_size,
                train_epochs=self.train_epochs,
                index_trees=self.index_trees,
                texts=self.texts,
                ids=self.ids)

    def set_params_dict(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def fit(self, texts, ids):
        train_corpus = list(tag_docs(texts, ids))

        doc2vec = Doc2Vec(vector_size=self.vector_size,
                            min_count=2,
                            epochs=self.train_epochs,
                            workers=4)

        doc2vec.build_vocab(train_corpus)

        class EpochLogger(CallbackAny2Vec):
            def __init__(self):
                self.epoch = 0

            def on_epoch_end(self, model):
                self.epoch += 1
                if self.epoch % 10 == 0:
                    print(f"Epoch #{self.epoch}")

        doc2vec.train(train_corpus, 
            total_examples=doc2vec.corpus_count,
            epochs=doc2vec.epochs,
            callbacks=[EpochLogger()])

        indexer = AnnoyIndexer(doc2vec, self.index_trees)

        self.doc2vec = doc2vec
        self.indexer = indexer

    def lookup(self, text, n=10):
        vector = self.doc2vec.infer_vector(tokenize_doc(text))
        neighboors = self.doc2vec.wv.most_similar([vector], topn=n, indexer=self.indexer)
        return neighboors

    def lookup_ids(self, text, n=10):
        neighboors = self.lookup(text, n=n)
        return [n[0] for n in neighboors]

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        pickle.dump(self.get_params_dict(), open( os.path.join(dir_path, 'emb.params'), "wb" ))
        self.doc2vec.save(os.path.join(dir_path, 'emb.doc2vec'))
        self.indexer.save(os.path.join(dir_path, 'emb.annoy'))

    @classmethod
    def load(cls, dir_path):
        params = pickle.load(open(os.path.join(dir_path, 'emb.params'), "rb" ))
        doc2vec = Doc2Vec.load(os.path.join(dir_path, 'emb.doc2vec'))
        indexer = AnnoyIndexer()
        indexer.load(os.path.join(dir_path, 'emb.annoy'))
        indexer.model = doc2vec
        emb = Embedder()
        emb.set_params_dict(params)
        emb.doc2vec = doc2vec
        emb.indexer = indexer
        return emb
