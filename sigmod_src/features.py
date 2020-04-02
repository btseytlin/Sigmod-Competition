import os
import sys
import random
import re
import numpy as np
import pandas as pd
from numba import jit, njit, prange
import Levenshtein as lev
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

def make_tfidf_features(df, vectorizers=None, tfidf_args=None):
    tfidf_args = tfidf_args or dict(
                strip_accents='ascii',
                min_df=2,
                max_df=0.95,
                max_features=500,
                ngram_range=(1,1)
            )

    vectorizers = vectorizers or {

    }

    fields_to_transform = [
        'page_title_stem'
    ]

    new_df = None
    for field in fields_to_transform:
        # For each field type we use it's own vectorizer
        vectorizer = vectorizers.get(field)
        if not vectorizer:
            vectorizer = TfidfVectorizer(**tfidf_args)

            vectorizer.fit(df[field])
            vectorizers[field] = vectorizer

        # Now transform
        tfidf = vectorizer.transform(df[field])
        tfidf_df = pd.DataFrame(tfidf.toarray(), 
            columns=[field+'__'+name for name in vectorizer.get_feature_names()])

        if new_df is not None:
            new_df = pd.concat([new_df, tfidf_df], axis=1)
        else:
            new_df = tfidf_df

    new_df.index = df.index
    return new_df, vectorizers


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
        if not left or not right or left =='n/a' or right == 'n/a':
            sums.append(0)
            continue
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
def common_symbols_normed(left_strings, right_strings):
    sums = np.zeros(len(left_strings))
    for idx in prange(len(left_strings)):
        left = left_strings[idx]
        right = right_strings[idx]
        if not left or not right or left =='n/a' or right == 'n/a':
            sums[idx] = 0
            continue

        running_sum = 0
        for i in range(len(left)):
            if i == len(right):
                break
            if left[i] != right[i]:
                break
            running_sum += 1
        running_sum = running_sum / (len(left)+len(right))/2
        sums[idx] = running_sum
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

