# -*- coding: utf-8 -*-
import os
import click
import logging

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import collections
import string
import nltk
import re
from gensim.utils import simple_preprocess

from ..utils import (read_json, path_from_spec_id,
 extract_brand, extract_site, extract_text)

nltk.download('punkt')


def make_classes_df(df, start_from_class=0):
    dup_df = df[df.label==1][['left_spec_id', 'right_spec_id', 'label']].sort_values(by=['left_spec_id', 'right_spec_id'])
    
    class_mapping = {

    }

    def replace_class_mapping(prev_class, new_class):
        for k in class_mapping:
            if k == prev_class:
                class_mapping[k] = new_class

    cur_class = start_from_class
    for row in dup_df.itertuples():
        if row.left_spec_id in class_mapping or row.right_spec_id in class_mapping:
            if row.left_spec_id in class_mapping and row.right_spec_id in class_mapping and class_mapping[row.left_spec_id] != class_mapping[row.right_spec_id]:
                raise Exception('wtf')

            existing_class = class_mapping.get(row.left_spec_id, class_mapping.get(row.right_spec_id))
            class_mapping[row.left_spec_id] = existing_class
            class_mapping[row.right_spec_id] = existing_class
        else:
            class_mapping[row.left_spec_id] = cur_class
            class_mapping[row.right_spec_id] = cur_class
            cur_class += 1

    classes_df = pd.DataFrame({'spec_id': list(class_mapping.keys()), 'class_': list(class_mapping.values())})
    return classes_df


def make_specs_dataset(specs_path):
    site_folders = os.listdir(specs_path)
    Row = collections.namedtuple('Row', ['spec_id', 'page_title', 'all_text'])
    rows = []
    for site in site_folders:
        for fname in os.listdir(os.path.join(specs_path, site)):
            path = os.path.join(specs_path, site, fname)
            parsed = read_json(path)
            all_text = '\t'.join(extract_text(parsed))
            row = Row(site+'//'+fname.split('.')[0], parsed['<page title>'], all_text)
            rows.append(row)
    specs_df = pd.DataFrame(rows)
    return specs_df


def hard_replaces(text):
    text = text.replace('| eBay', '')\
                            .replace("Cannon", 'canon')
    text = re.sub(r'(\d+) x (\d+)', r'\1x\2', text)
    text = re.sub(r'\d{8,}', '', text)
    return text


def get_unfrequent(texts, cutoff=0):
    """
        Returns all tokens that appear in <= than `cutoff` texts
    """

    word_counts = collections.Counter()
    for phrase in texts:
        for word in set(phrase.split()):
            word_counts[word] += 1

    return set([k for k, v in word_counts.items() if v <= cutoff])


def preprocess_text_field(text, unfrequent=set(), max_words=200):
    nltk_stopswords = nltk.corpus.stopwords.words('english')
    stopwords=set(list(nltk_stopswords)+['camera',
               'product',
               'used', 'black', 'white',
              'reviews', 'price', 'digital', 'slr', 'digital slr', 'dslr',
              'new', 'used', 'brand', 'buy' ,'item', 'listing', 
              'condition', 'see'])

    printable = set(string.printable)

    text = hard_replaces(text)
    text = text.lower()
    text = text.encode("ascii", errors="ignore").decode() #remove non ascii
    text = ''.join([w for w in text if w in printable]).strip()
    
    text = ' '.join([w for w in nltk.tokenize.word_tokenize(text)
        if not w in stopwords 
        and w.isalnum()
        and not w in unfrequent
        ][:max_words])
    return text

def preprocess_text_column(texts, cutoff=0):
    unfrequent = get_unfrequent(texts, cutoff=cutoff)
    new_texts = texts.apply(lambda x: preprocess_text_field(x, unfrequent))
    return new_texts

def preprocess_specs_dataset(specs_df):
    """
        Clean page titles, fix known typos, apply stemming
    """
    printable = set(string.printable)
    snow = nltk.stem.SnowballStemmer('english')
            
    def stem(text):
        return ' '.join([snow.stem(w) for w in text.split(' ')]).strip()

    # Clean up
    specs_df['page_title'] = preprocess_text_column(specs_df.page_title)

    specs_df['all_text'] = preprocess_text_column(specs_df.all_text, cutoff=1)

    # Stem
    specs_df['page_title_stem'] = specs_df.page_title.apply(stem)

    specs_df['all_text_stem'] = specs_df.all_text.apply(stem)

    specs_df['brand'] = specs_df.all_text.apply(extract_brand)

    specs_df['site'] = specs_df.spec_id.apply(extract_site)

    # Drop null titles
    bad_row_selector = (specs_df.page_title.isnull()) | (specs_df.page_title=='') | (specs_df.page_title=='null')
    null_rows = specs_df[bad_row_selector].shape[0]
    if null_rows:
        specs_df = specs_df[~bad_row_selector]
        print(f'Warning, dropped {null_rows} rows containing null page titles')
    return specs_df


def make_labelled_dataset(labels_path, specs_df):
    """
        join specs_df to labels
    """
    labels_df = pd.read_csv(labels_path)

    specs_df = specs_df.drop(['page_title', 
                              'page_title_stem',
                              'brand',
                              'site'], axis=1)
    sides = ['left', 'right']
    for side in sides:
        join_df = specs_df.copy()
        join_df.columns = [side+'_'+col for col in join_df.columns]
        labels_df = labels_df.merge(join_df, on=side+'_spec_id', how='left')
    return labels_df


def make_and_write_dataset(labels_path, specs_path, out_path='../data/interim/dataset.csv'):
    df = make_dataset(labels_path, specs_path)
    df.to_csv(out_path)


@click.command()
@click.argument('labels_path', type=click.Path(exists=True))
@click.argument('specs_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    make_and_write_dataset(labels_path, specs_path, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
