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

from ..utils import read_json, path_from_spec_id, extract_brand, extract_site

def make_specs_dataset(specs_path):
    site_folders = os.listdir(specs_path)
    Row = collections.namedtuple('Row', ['spec_id', 'page_title'])
    rows = []
    for site in site_folders:
        for fname in os.listdir(os.path.join(specs_path, site)):
            path = os.path.join(specs_path, site, fname)
            parsed = read_json(path)
            row = Row(site+'//'+fname.split('.')[0], parsed['<page title>'])
            rows.append(row)
    specs_df = pd.DataFrame(rows)
    return specs_df


def preprocess_specs_dataset(specs_df):
    """
        Clean page titles, fix known typos, apply stemming
    """
    printable = set(string.printable)
    snow = nltk.stem.SnowballStemmer('english')

    def clean(page_title):
        page_title = page_title.replace('| eBay', '').replace("Cannon", 'canon')
        page_title = page_title.encode("ascii", errors="ignore").decode() #remove non ascii
        page_title = ''.join([w for w in page_title if w in printable]).strip()
        return page_title
            
    def stem(page_title):
        return ' '.join([snow.stem(w) for w in page_title.split(' ')]).strip()

    # Clean up
    specs_df['page_title'] = specs_df.page_title.apply(clean)

    # Stem
    specs_df['page_title_stem'] = specs_df.page_title.apply(stem)

    specs_df['brand'] = specs_df.page_title.apply(extract_brand)

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
