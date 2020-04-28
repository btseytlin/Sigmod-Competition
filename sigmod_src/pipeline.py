import os
import sys
import random
import re
from itertools import combinations
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
from numba import jit, prange
from sklearn import preprocessing
from .utils import extract_special_tokens, extract_number_tokens, get_additional_labels, make_graph_or_load
from .features import (make_tfidf_features, get_common_tokens, get_sum_len_n_common, pairwise_cosine_dist, 
    pairwise_jaccard, common_symbols_from_start, common_symbols_normed, levenstein, n_graph_common_neighboors)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier 
from sklearn.ensemble import AdaBoostClassifier


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
        self.specs_df = specs_df.copy()
        self.specs_df['spec_idx'] = range(len(self.specs_df))
        self.labels_df = labels_df

        self.additional_df = None

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

    def __init__(self, specs_df, labels_df, 
                 submit_fpath='../data/submit/submit.csv',
                 submit_batch_size=10000, additional_label_ratio=1,
                 graph_fpath='../data/processed/graph_edgelist.txt'):
        super().__init__(specs_df, labels_df, submit_fpath, submit_batch_size)

        self.graph_fpath = graph_fpath
        self.additional_label_ratio = additional_label_ratio
        self.train_X = None

        self.feature_names = ['n_common_tokens', 
                    'n_common_tokens_normed',
                    'sum_len_common_tokens',
                    'special_sum_len_common_tokens',
                    'special_n_common_tokens',
                    'number_sum_len_common_tokens',
                    'number_n_common_tokens',

                    'n_common_symbols_models', 'same_model',

                    'n_common_symbols_types', 'same_type',

                    'n_common_symbols_megapixels', 'same_megapixels',

                    #'cosine_sim_tfidf',
                    'lev_ratios',

                    'n_common_neighboors', 'n_common_neighboors_normed',

                    'jaccard_sim',
                    'n_common_symbols',


                    'site_left', 'site_right', 


                    'sum_len_common_tokens_all_text', 'n_common_tokens_all_text',
                    'special_n_common_tokens_all_text', 'special_n_common_tokens_all_text_normed',


                   'same_site',
                   ]

    def precompute(self):
        """Pre-computes tfidf vectors for specs, label encodes site and brand"""
        self.additional_df = get_additional_labels(self.labels_df, self.specs_df)

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

        self.spec_special_tokens = self.specs_df.page_title_stem.apply(extract_special_tokens).values
        self.spec_number_tokens = self.specs_df.page_title_stem.apply(extract_number_tokens).values

        self.spec_models = self.specs_df.model.fillna('n/a').values

        self.spec_types = self.specs_df.type.fillna('n/a').values
        self.spec_megapixels = self.specs_df.megapixels.fillna('n/a').values

        #

        self.spec_tokens_all_text = self.specs_df.all_text_stem.str.split(' ').values
        self.spec_special_tokens_all_text = self.specs_df.all_text_stem.apply(extract_special_tokens).values

        self.graph = make_graph_or_load(self.specs_df, self.graph_fpath)

        adj = self.graph.adj
        neighboors = []
        for i in range(len(self.specs_df)):
            if not i in adj:
                neighboors.append(set())
            else:
                neighboors.append(set(adj[i]))

        self.neighboors = np.array(neighboors)

        
    def make_X(self, left_idx, right_idx):
        left_titles = self.spec_titles[left_idx]
        right_titles = self.spec_titles[right_idx]

        left_models = self.spec_models[left_idx]
        right_models = self.spec_models[right_idx]

        left_types = self.spec_types[left_idx]
        right_types = self.spec_types[right_idx]

        left_megapixels = self.spec_megapixels[left_idx]
        right_megapixels = self.spec_megapixels[right_idx]

        left_tokens = self.spec_tokens[left_idx]
        right_tokens = self.spec_tokens[right_idx]

        # print("Getting common tokens features")
        token_pairs = list(zip(left_tokens, right_tokens))
        common_tokens, n_total_tokens = get_common_tokens(token_pairs)
        sum_len_common_tokens, n_common_tokens = get_sum_len_n_common(common_tokens)

        sum_len_common_tokens = np.array(sum_len_common_tokens)
        n_common_tokens = np.array(n_common_tokens)
        n_common_tokens_normed = n_common_tokens/np.array(n_total_tokens)

        # print('Getting special tokens features')
        special_left_tokens = self.spec_special_tokens[left_idx]
        special_right_tokens = self.spec_special_tokens[right_idx]

        special_token_pairs = list(zip(special_left_tokens, special_right_tokens))
        special_common_tokens, _ = get_common_tokens(special_token_pairs)
        special_sum_len_common_tokens, special_n_common_tokens = get_sum_len_n_common(special_common_tokens)
        special_sum_len_common_tokens = np.array(special_sum_len_common_tokens)
        special_n_common_tokens = np.array(special_n_common_tokens)
        special_n_common_tokens_normed = special_n_common_tokens / np.array(n_total_tokens)

        #print('Getting features for all text')

        left_tokens_all_text = self.spec_tokens_all_text[left_idx]
        right_tokens_all_text = self.spec_tokens_all_text[right_idx]

        special_left_tokens_all_text = self.spec_special_tokens_all_text[left_idx]
        special_right_tokens_all_text = self.spec_special_tokens_all_text[right_idx]

        token_pairs_all_text = list(zip(left_tokens_all_text, right_tokens_all_text))
        common_tokens_all_text, n_total_tokens_all_text = get_common_tokens(token_pairs_all_text)
        sum_len_common_tokens_all_text, n_common_tokens_all_text = get_sum_len_n_common(common_tokens_all_text)

        sum_len_common_tokens_all_text = np.array(sum_len_common_tokens_all_text)
        n_common_tokens_all_text = np.array(n_common_tokens_all_text)
        n_common_tokens_normed_all_text = n_common_tokens_all_text/np.array(n_total_tokens_all_text)
        

        special_token_pairs_all_text = list(zip(special_left_tokens_all_text, special_right_tokens_all_text))
        special_common_tokens_all_text, _ = get_common_tokens(special_token_pairs_all_text)
        special_sum_len_common_tokens_all_text, special_n_common_tokens_all_text = get_sum_len_n_common(special_common_tokens_all_text)
        special_sum_len_common_tokens_all_text = np.array(special_sum_len_common_tokens_all_text)
        special_n_common_tokens_all_text = np.array(special_n_common_tokens_all_text)
        special_n_common_tokens_all_text_normed = special_n_common_tokens_all_text / np.array(n_total_tokens_all_text)

        #print('Getting number tokens features')
        number_left_tokens = self.spec_number_tokens[left_idx]
        number_right_tokens = self.spec_number_tokens[right_idx]

        number_token_pairs = list(zip(number_left_tokens, number_right_tokens))
        number_common_tokens, _ = get_common_tokens(number_token_pairs)
        number_sum_len_common_tokens, number_n_common_tokens = get_sum_len_n_common(number_common_tokens)
        number_sum_len_common_tokens = np.array(number_sum_len_common_tokens)
        number_n_common_tokens = np.array(number_n_common_tokens)
        number_n_common_tokens_normed = number_n_common_tokens / np.array(n_total_tokens)


        # print("Getting TFIDF cosine")
        # tfidf_left, tfidf_right = self.tfidf.values[left_idx], self.tfidf.values[right_idx]
        # norms_left, norms_right = self.tfidf_norms[left_idx], self.tfidf_norms[right_idx]
        

        # cosine_sim_tfidf = pairwise_cosine_dist(tfidf_left, tfidf_right, norms_left, norms_right)

        #print("Getting Jaccard")
        

        jaccard_sim = np.array(pairwise_jaccard(token_pairs))

        #print("Getting Levenstein")

        

        lev_ratios = levenstein(left_titles, right_titles)
        lev_ratios = np.array(lev_ratios)

        # Слишком медленно
        # print("Getting lcs")
        # lcs = np.array(pairwise_lcs(left_titles, right_titles))

        # print("Getting common symbols")
        n_common_symbols = np.array(common_symbols_from_start(left_titles, right_titles))

        # print('Getting common symbols in models')
        n_common_symbols_models = np.array(common_symbols_normed(left_models, right_models))
        same_model = np.array((left_models == right_models)).astype(int)


        # print('Getting common symbols in types')

        n_common_symbols_types = np.array(common_symbols_normed(left_types, right_types))
        same_type = np.array((left_types == right_types) & (left_types != 'n/a') & (right_types != 'n/a')).astype(int)

        #print('Getting common symbols in megapixels')
        n_common_symbols_megapixels = np.array(common_symbols_normed(left_megapixels, right_megapixels))
        same_megapixels = np.array((left_megapixels == right_megapixels) & (left_megapixels != 'n/a') & (right_megapixels != 'n/a')).astype(int)


        #print('Computing graph features')
        left_neighboors, right_neighboors = self.neighboors[left_idx], self.neighboors[right_idx]  
        n_common_neighboors, n_common_neighboors_normed = n_graph_common_neighboors(left_neighboors, right_neighboors)


        site_left = self.site_enc[left_idx]
        site_right = self.site_enc[right_idx]

        # brand_left = self.brand_enc[left_idx]
        # brand_right = self.brand_enc[right_idx]

        # same_brand = np.array(brand_left == brand_right).astype(int)

        same_site = np.array(site_left == site_right).astype(int)

        features = [n_common_tokens, 
                    n_common_tokens_normed,
                    sum_len_common_tokens,
                    special_sum_len_common_tokens,
                    special_n_common_tokens,
                    number_sum_len_common_tokens,
                    number_n_common_tokens,

                    n_common_symbols_models, same_model,

                    n_common_symbols_types, same_type,

                    n_common_symbols_megapixels, same_megapixels,

                    # cosine_sim_tfidf,
                    lev_ratios,

                    # n_common_neighboors, n_common_neighboors_normed,

                    jaccard_sim,
                    n_common_symbols,

                    #n_common_neighboors, n_common_neighboors_normed,

                    site_left, site_right, 

                    #brand_left, brand_right, same_brand,

                    sum_len_common_tokens_all_text, n_common_tokens_all_text,
                    special_n_common_tokens_all_text, special_n_common_tokens_all_text_normed,


                   same_site,
                   ]
        
        return np.hstack([np.array(f).reshape(-1, 1) if len(f.shape)==1 else f for f in features])

    def train(self, precompute=True):
        print('Precomputing')
        if precompute:
            self.precompute()

        self.clf = VotingClassifier([
                        ('lgb1', LGBMClassifier(sample_pos_weight=5.76)),
                        ('lgb2', LGBMClassifier(sample_pos_weight=5.76, n_estimators=500)),
                        ('lgb3', LGBMClassifier(sample_pos_weight=5.76, learning_rate=0.01)),
                        ('lgb4', LGBMClassifier(sample_pos_weight=5.76, learning_rate=0.01, n_estimators=500)),
                        ('logreg', Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])),
                        ('gb', Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())])),
                        ('mlp', Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier())])),
                    ], n_jobs=-1)

        print('Making features')
        left_spec_idxs = self.specs_id_to_idx[self.labels_df['left_spec_id']]
        right_spec_idxs = self.specs_id_to_idx[self.labels_df['right_spec_id']]

        left_brand, right_brand = self.brand_enc[left_spec_idxs], self.brand_enc[right_spec_idxs]
        match_index = np.argwhere(left_brand == right_brand).flatten()
        left_spec_idxs = left_spec_idxs[match_index]
        right_spec_idxs = right_spec_idxs[match_index]

        X = self.make_X(left_spec_idxs, right_spec_idxs)
        self.train_X = X
        self.train_Y = self.labels[match_index]
        assert self.train_X.shape[0] == self.train_Y.shape[0]
        print('Making features for additional_labels')

        additional_df = self.additional_df.sample(int(len(self.labels_df)*self.additional_label_ratio))

        left_spec_idxs = self.specs_id_to_idx[additional_df['left_spec_id']]
        right_spec_idxs = self.specs_id_to_idx[additional_df['right_spec_id']]

        left_brand, right_brand = self.brand_enc[left_spec_idxs], self.brand_enc[right_spec_idxs]
        match_index = np.argwhere(left_brand == right_brand).flatten()
        left_spec_idxs = left_spec_idxs[match_index]
        right_spec_idxs = right_spec_idxs[match_index]

        self.additional_X = self.make_X(left_spec_idxs, right_spec_idxs)
        self.additional_Y = additional_df.label.values[match_index]
        assert self.additional_X.shape[0] == self.additional_Y.shape[0]
        full_X = np.vstack([self.train_X, self.additional_X])
        full_Y = np.hstack([self.train_Y, self.additional_Y]).flatten()

        print('Fitting model')
        self.clf.fit(full_X, full_Y)



    def make_submission(self):
        if os.path.exists(self.submit_fpath):
            os.remove(self.submit_fpath)

        # Remove specs present in labels_df
        labelled_specs = set(self.labels_df.left_spec_id).union(self.labels_df.right_spec_id)
        oof_specs_df = self.specs_df[~self.specs_df.spec_id.isin(labelled_specs)].copy()

        oof_specs_df['brand'] = oof_specs_df.brand.fillna('missing')
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
