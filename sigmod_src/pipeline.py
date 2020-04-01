import os
import sys
import random
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn import preprocessing
from lightgbm import LGBMClassifier
from sigmod_src.features.build_features import make_tfidf_features
from tqdm import trange, tqdm_notebook as tqdm
from scipy.spatial.distance import cdist
from numba import vectorize, jit, njit, prange
import Levenshtein as lev


@jit(parallel=True)
def get_common_tokens(token_pairs):
    common_tokens = []
    for i in prange(len(token_pairs)):
        left, right = token_pairs[i][0], token_pairs[i][1]
        common_tokens.append(list(set(left).intersection(right)))
    return common_tokens

@jit(parallel=True)
def get_sum_len_n_common(common_tokens):
    n_common = []
    sum_lens = []
    for i in prange(len(common_tokens)):
        n_common.append(len(common_tokens[i]))
        running_sum = 0
        for c in common_tokens[i]:
            running_sum += len(c)
        sum_lens.append(running_sum)
    return sum_lens, n_common

@jit(target='cpu', nopython=True, parallel=True)
def pairwise_cosine_dist(tfidf_left, tfidf_right, norms_left, norms_right):
    cosine_sims = np.zeros(len(tfidf_left))
    for i in prange(len(tfidf_left)):
        norm_left = norms_left[i]
        norm_right = norms_right[i]
        cosine_sims[i] = (tfidf_left[i, :] @ tfidf_right[i, :]) / (norm_left * norm_right)
    return cosine_sims

@jit(parallel=True)
def pairwise_jaccard(token_pairs):
    jaccard_sims = []
    for i in prange(len(token_pairs)):
        left, right = set(token_pairs[i][0]), set(token_pairs[i][1])
        jaccard_sims.append(len(left.intersection(right)) / len(left.union(right)))

    return jaccard_sims

def longest_common_substring(a, b):
    # generate matrix of length of longest common subsequence for substrings of both words
    lengths = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
 
    # read a substring from the matrix
    res_len = 0
    result = ''
    j = len(b)
    for i in range(1, len(a)+1):
        if lengths[i][j] != lengths[i-1][j]:
            result += a[i-1]
            res_len += 1
    return res_len


@jit(parallel=True)
def pairwise_lcs(left_strings, right_strings):
    dists = []
    for i in prange(len(left_strings)):
        a = left_strings[i]
        b = right_strings[i]
        dists.append(longest_common_substring(a, b))
    return dists


@jit(parallel=True)
def common_symbols_from_start(left_strings, right_strings):
    sums = []
    for i in prange(len(left_strings)):
        left = left_strings[i]
        right = right_strings[i]
        running_sum = 0
        for i in range(len(left)):
            if i == len(right):
                break
            if left[i] != right[i]:
                break
            running_sum += 1
        sums.append(running_sum)
    return sums
         

@jit(parallel=True)
def levenstein(left_strings, right_strings):
    distances = []
    ratios = []
    for i in prange(len(left_strings)):
        left = left_strings[i]
        right = right_strings[i]

        distances.append(lev.distance(left, right))
        ratios.append(lev.ratio(left, right))

    return distances, ratios

class BasePipeline:
    """Base class that encapsulates all stages of the submission process:

       1. Takes in preprocessed specs data frame, labels dataframe. 
       2. Precomputes stuff for features
       3. Creates features for a training dataset
       4. Trains a model
       5. Generates a submission by going over all specs

    """
    def __init__(self, specs_df, labels_df, 
                 submit_fpath='../data/submit/submit.csv',
                 submit_batch_size=10000):
        self.specs_df = specs_df
        self.specs_df['spec_idx'] = range(len(self.specs_df))
        self.labels_df = labels_df

        self.submit_fpath = submit_fpath
        self.submit_batch_size = submit_batch_size

        self.spec_ids = specs_df.spec_id.values
        self.specs_id_to_idx = pd.Series({v: k for k, v in enumerate(specs_df.spec_id.values)})
        self.specs = specs_df.values
        self.labels = labels_df.label.values

        self.clf = None

    def precompute(self):
        pass

    def train(self):
        pass

    def make_submission(self):
        pass

class LGBMPipeline(BasePipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def precompute(self):
        """Pre-computes tfidf vectors for specs, label encodes site and brand"""
        # Compute tfidf
        self.tfidf, self.vectorizers = make_tfidf_features(self.specs_df)
        self.tfidf_norms = []
        for i in range(len(self.tfidf)):
            self.tfidf_norms.append(np.linalg.norm(self.tfidf.values[i, :], 2))
        self.tfidf_norms = np.array(self.tfidf_norms)

        # Label-encode categorical
        site = self.specs_df.site.fillna('n/a')
        brand = self.specs_df.brand.fillna('n/a')

        site_le = preprocessing.LabelEncoder()
        site_le.fit(site)
        self.site_enc = site_le.transform(site)

        brand_le = preprocessing.LabelEncoder()
        brand_le.fit(brand)
        self.brand_enc = brand_le.transform(brand)

        self.label_encoders = {'site': site_le, 'brand': brand_le}

        self.spec_titles = self.specs_df.page_title_stem.values
        self.spec_tokens = self.specs_df.page_title_stem.str.split(' ').values

        
    def make_X(self, left_idx, right_idx):
        left_titles = self.spec_titles[left_idx]
        right_titles = self.spec_titles[right_idx]

        left_tokens = self.spec_tokens[left_idx]
        right_tokens = self.spec_tokens[right_idx]

        # print("Getting common tokens features")
        token_pairs = list(zip(left_tokens, right_tokens))
        common_tokens = get_common_tokens(token_pairs)
        sum_len_common_tokens, n_common_tokens = get_sum_len_n_common(common_tokens)
        sum_len_common_tokens = np.array(sum_len_common_tokens)
        n_common_tokens = np.array(n_common_tokens)

        # print("Getting TFIDF cosine")
        tfidf_left, tfidf_right = self.tfidf.values[left_idx], self.tfidf.values[right_idx]
        norms_left, norms_right = self.tfidf_norms[left_idx], self.tfidf_norms[right_idx]
        

        cosine_sim = pairwise_cosine_dist(tfidf_left, tfidf_right, norms_left, norms_right)

        # print("Getting Jaccard")
        

        jaccard_sim = np.array(pairwise_jaccard(token_pairs))

        # print("Getting Levenstein")

        

        lev_distances, lev_ratios = levenstein(left_titles, right_titles)
        lev_distances = np.array(lev_distances)
        lev_ratios = np.array(lev_ratios)

        # Слишком медленно
        # print("Getting lcs")
        # lcs = np.array(pairwise_lcs(left_titles, right_titles))

        # print("Getting common symbols")
        n_common_symbols = np.array(common_symbols_from_start(left_titles, right_titles))

        site_left = self.site_enc[left_idx]
        site_right = self.site_enc[right_idx]

        brand_left = self.brand_enc[left_idx]
        brand_right = self.brand_enc[right_idx]

        same_brand = brand_left == brand_right
        same_site = site_left == site_right


        features = [n_common_tokens, 
                    sum_len_common_tokens,
                    jaccard_sim,
                    cosine_sim, 
                    lev_distances, lev_ratios,
                    n_common_symbols,
                    site_left, site_right, brand_left, brand_right,
                   same_brand, same_site]
        
        
        return np.hstack([f.reshape(-1, 1) if len(f.shape)==1 else f for f in features])

    def train(self, precompute=True):
        print('Precomputing')
        if precompute:
            self.precompute()

        self.clf = LGBMClassifier(sample_pos_weight=5.76,
                             n_jobs=-1)

        print('Making features')
        left_spec_idxs = self.specs_id_to_idx[self.labels_df['left_spec_id']]
        right_spec_idxs = self.specs_id_to_idx[self.labels_df['right_spec_id']]
        X = self.make_X(left_spec_idxs, right_spec_idxs)

        print('Fitting model')
        self.clf.fit(X, self.labels)



    def make_submission(self):
        if os.path.exists(self.submit_fpath):
            os.remove(self.submit_fpath)

        # Remove specs present in labels_df
        labelled_specs = set(self.labels_df.left_spec_id).union(self.labels_df.right_spec_id)
        oof_specs_df = self.specs_df[~self.specs_df.spec_id.isin(labelled_specs)]


        brand_groups = oof_specs_df.groupby('brand')['spec_idx'].agg(list).to_dict()

        @jit(parallel=True)
        def process_group(brand, group_specs, batch_size):
            brand_combs = np.array(list(combinations(group_specs, 2)))
            for i in prange(0, len(brand_combs), batch_size):
                batch_left_spec_idxs = brand_combs[i:i+batch_size][:, 0]
                batch_right_spec_idxs = brand_combs[i:i+batch_size][:, 1]
                
                batch_left_spec_ids = self.spec_ids[batch_left_spec_idxs]
                batch_right_spec_ids = self.spec_ids[batch_right_spec_idxs]
                
                id_pairs = np.column_stack([batch_left_spec_ids, batch_right_spec_ids])
                X = self.make_X(batch_left_spec_idxs, batch_right_spec_idxs)
                labels = self.clf.predict(X)

                dup_idx = np.argwhere(labels==1)
                dup_pairs = id_pairs[dup_idx.flatten(), :]

                if not dup_pairs.any():
                    continue
                out_df = pd.DataFrame(dup_pairs, columns=['left_spec_id', 'right_spec_id'])
                if not out_df.empty:
                    if os.path.exists(self.submit_fpath):
                        out_df.to_csv(self.submit_fpath, mode='a', header=False, index=False)
                    else:
                        out_df.to_csv(self.submit_fpath, index=False)

        batch_size = self.submit_batch_size
        for brand, group_specs in tqdm(brand_groups.items()):
            process_group(brand, group_specs, batch_size)
