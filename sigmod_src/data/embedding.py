import gensim
from gensim.similarities.index import AnnoyIndexer
from sklearn.model_selection import train_test_split

def tokenize_doc(doc):
    return doc.split(' ')

def tokenize_docs(docs):
    for doc in docs:
        yield tokenize_doc(doc)

def tag_docs(docs, ids):
    for doc_id, doc in zip(ids, docs):
        tokens = tokenize_doc(doc)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [doc_id])


def get_embedder(texts, ids, vector_size=200, epochs=100, trees=100):
    train_corpus = list(tag_docs(texts, ids))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size,
        min_count=2,
        epochs=epochs,
        workers=4)

    model.build_vocab(train_corpus)

    model.train(train_corpus, 
        total_examples=model.corpus_count,
        epochs=model.epochs)

    indexer = AnnoyIndexer(model, trees)

    return model, indexer

def emb_lookup(text, emb, indexer, n=10):
    vector = emb.infer_vector(tokenize_doc(text))
    neighboors = emb.wv.most_similar([vector], topn=n, indexer=indexer)
    return neighboors