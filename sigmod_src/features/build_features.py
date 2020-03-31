import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

def make_tfidf_features(df, vectorizers=None, tfidf_args=None):
    tfidf_args = tfidf_args or dict(
                strip_accents='ascii',
                min_df=2,
                max_df=0.95,
                max_features=500,
                ngram_range=(1,2)
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

def make_categorical_features(df, site_le=None, brand_le=None):
    if not site_le:
        site_le = preprocessing.LabelEncoder()
        site_le.fit(df.site)

    site_enc = site_le.transform(df.site)

    if not brand_le:
        brand_le = preprocessing.LabelEncoder()
        brand_le.fit(df.brand)

    brand_enc = brand_le.transform(df.brand)

    return pd.DataFrame({'site_enc': site_enc, 'brand_enc': brand_enc}, 
            index=df.index), site_le, brand_le


def make_features(specs_df, vectorizers=None,
                        site_le=None,
                        brand_le=None):
    """
        specs_df: preprocessed specs_df
    """
    tfidf_df, vectorizers = make_tfidf_features(specs_df, vectorizers)
    categorial_df, site_le, brand_le = make_categorical_features(specs_df, site_le)
    features_df = tfidf_df.join(categorial_df)

    assert features_df.shape[0] == specs_df.shape[0]
    features_df['spec_id'] = specs_df.spec_id
    return features_df, vectorizers, {'site': site_le, 'brand': brand_le}

