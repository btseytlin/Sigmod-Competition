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
                 submit_batch_size=5000):
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

        self.spec_tokens = self.specs_df.page_title_stem.str.split(' ').values
        
        
    def make_X(self, left_idx, right_idx):
        left_tokens = self.spec_tokens[left_idx]
        right_tokens = self.spec_tokens[right_idx]

        n_common_tokens = np.array([len(np.intersect1d(left_tokens[i], right_tokens[i])) for i in range(len(left_tokens))])

        tfidf_left, tfidf_right = self.tfidf.values[left_idx], self.tfidf.values[right_idx]
        # cosine_sim = cdist(tfidf_left, tfidf_right)
        site_left = self.site_enc[left_idx]
        site_right = self.site_enc[right_idx]

        brand_left = self.brand_enc[left_idx]
        brand_right = self.brand_enc[right_idx]

        same_brand = brand_left == brand_right
        same_site = site_left == site_right


        features = [n_common_tokens, 
                    # cosine_sim, 
                    site_left, site_right, brand_left, brand_right,
                   same_brand, same_site]
        
        
        return np.hstack([f.reshape(-1, 1) if len(f.shape)==1 else f for f in features])

    def train(self, precompute=True):
        if precompute:
            self.precompute()

        self.clf = LGBMClassifier(sample_pos_weight=5.76,
                             n_jobs=-1)

        left_spec_idxs = self.specs_id_to_idx[self.labels_df['left_spec_id']]
        right_spec_idxs = self.specs_id_to_idx[self.labels_df['right_spec_id']]
        X = self.make_X(left_spec_idxs, right_spec_idxs)

        self.clf.fit(X, self.labels)



    def make_submission(self):
        if os.path.exists(self.submit_fpath):
            os.remove(self.submit_fpath)

        # Remove specs present in labels_df
        labelled_specs = set(self.labels_df.left_spec_id).union(self.labels_df.right_spec_id)
        oof_specs_df = self.specs_df[~self.specs_df.spec_id.isin(labelled_specs)]


        brand_groups = oof_specs_df.groupby('brand')['spec_idx'].agg(list).to_dict()

        batch_size = self.submit_batch_size
        for brand, group_specs in tqdm(brand_groups.items()):
            brand_combs = np.array(list(combinations(group_specs, 2)))
            for i in tqdm(range(0, len(brand_combs), batch_size)):
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
