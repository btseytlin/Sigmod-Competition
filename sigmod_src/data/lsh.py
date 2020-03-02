from sklearn.preprocessing import normalize
from tqdm import trange, tqdm_notebook as tqdm
from annoy import AnnoyIndex

def load_index(path, vector_dim):
    lsh = AnnoyIndex(vector_dim, 'angular')
    lsh.load(path) 
    return lsh

def make_index(vectors, trees_amount):
    vectors = normalize(vectors)
    vector_dim = vectors.shape[1]
    
    lsh = AnnoyIndex(vector_dim, 'angular')
    for i in trange(len(vectors)):
        lsh.add_item(i, vectors[i])
        
    lsh.build(trees_amount)

    return lsh, vector_dim

def query_by_vector(lsh, v, n=10, **kwargs):
    normalized = normalize([v])[0]
    return lsh.get_nns_by_vector(normalized, n+1, **kwargs)

def query_id_by_vector(lsh, ids, v, n=10, **kwargs):
    nns_idx = query_by_vector(lsh, v, n=n, **kwargs)
    distances = None
    if isinstance(nns_idx, tuple):
        nns_idx, distances = nns_idx

    if distances:
        return ids[nns_idx], distances
    return ids[nns_idx]