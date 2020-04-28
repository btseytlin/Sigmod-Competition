import os
import re
import itertools
from tqdm import trange, tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import collections
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from numba import vectorize, jit, njit, prange
from .utils import (extract_site, special_token_pattern, get_known_items)

printable = set(string.printable)

major_camera_brands = ['vivitar', 'visiontek', 'vageeswari', 'traveler', 'thomson',
                       'tevion', 'samsung', 'rollei', 'ricoh', 'praktica', 
                       'polaroid', 'phase one', 'pentax', 'panasonic', 'olympus', 'nikon',
                       'minox', 'memoto', 'medion', 'lytro', 'leica', 'kodak', 'hp',
                       'hasselblad', 'genius', 'fujifilm', 'foscam', 
                       'epson', 'casio', 'canon', 'blackmagic design',
                       'benq', 'bell & howell', 'bell', 'aigo', 'agfaphoto', 'advert tech',
                       'dahua', 'philips', 'sanyo', 'vizio', 'sharp',
                       'logitech', 'hikvision', 'bell', 'topixo', 'magnavox',
                       'samyang', 'sekonic', 'lexar', 'ksm', 'uv', 'hoya', 'dahua',
                       'colorpix', 'onvif', 'sjcam', 'ge '
                      ]

brand_blacklist = ['shoot', 'as',  'eos', 'action', 'new', 'class', 'sharp', 'digital', 'sj4000', 'for', 'rca']

drop_brands = ['telesin',
               'carbose',
               'cxsin',
               'fantasea',
               'hikvision',
               'unbranded generic',
               'neopine', 'godspeed', 'opteka',
               'dahua']

def populate_category_from_all_text(all_text, item, field_name, known_items, blacklist, print_conflicts=False):
    """
      Example of item: brand

      Takes a text of spec, a known brand for spec and a list of known brands.
      
      If brand is None, goes over all known brands. If one is found in text, returns it as brand

      If brand is not none, but a different brand is found in text, returns brand

      If found brand and known brand for spec match, returns brand.

      If none is found and none is known, returns None.

    """
    if pd.isnull(item):
        item = None

    found_items = list(set([item for item in known_items if item in all_text]).difference(blacklist))
    found_items = [i.strip().lower() for i in found_items if i and i.strip().lower()]
    if not found_items:
        return item

    if not item:
        return found_items[0]

    conflict = False
    for found_item in found_items:
        if item and found_item:
            if found_item != item:
                conflict = True
            else:
                conflict = False
                return found_item

    if conflict and print_conflicts:
        print(f'Conflict, no matching items found for {field_name}:', item)
    return item

def populate_categories(all_texts, items, field_name, known_items, blacklist):
    new_items = []
    for i in prange(len(items)):
        all_text = all_texts[i]
        item = items[i]
        new_item = populate_category_from_all_text(all_text, item, field_name, known_items, blacklist)
        if new_item in blacklist:
            new_items.append(None)
        else:
            new_items.append(new_item)
    return new_items

def populate_models_from_all_text(all_text, item, known_items, blacklist):
    if item:
        return item

    found_items = known_items.intersection(all_text.split(' ')).difference(blacklist)
    found_items = [i.strip().lower() for i in found_items if i and i.strip().lower()]
    if found_items:
        for model in found_items:
            if model:
                return model
    return None


def populate_models(all_texts, items, known_items, blacklist):
    new_items = []
    for i in prange(len(items)):
        all_text = all_texts[i]
        item = items[i]
        new_item = populate_models_from_all_text(all_text, item, known_items, blacklist)
        new_items.append(new_item)
    return new_items

def hard_replaces(text):
    text = text.replace('| eBay', '')
    text = re.sub(r'\bcannon\b', r'canon', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfuji\b', r'fujifilm', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+) x (\d+)', r'\1x\2', text)
    text = text.replace(' +', ' ')

    text = re.sub(r'((\d+)[\s.]+){0,1}(\d+)[\s.]*((mp)|(megapixel))s?', r'\2-\3-mp', text, flags=re.IGNORECASE)

    text = re.sub(r'(\d) mm\b', r'\1mm ', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d)milimeter\b', r'\1mm ', text, flags=re.IGNORECASE)
    return text


def preprocess_text_field(text, unfrequent=None, max_words=500):
    unfrequent = unfrequent or set()
    if not text:
        return text
    nltk_stopswords = nltk.corpus.stopwords.words('english')
    stopwords=set(list(nltk_stopswords)+['camera',
               'product',
               'used', 'black', 'white',
              'reviews', 'price', 
              'new', 'used', 'brand', 'buy' ,'item', 'listing', 
              'condition', 'see',
              'white', 'black', 'silver', 'blue', 'red', 
              'grey', 'gray', 'read', 'more', 'about', 'unused', 'used',
              'opened', 'unopened', 'manufacturer', 'original', 'seller',
              'sellers', 'packaging', 'details', 'opens', 'damaged', 'undamaged',
              'window', 'tab', 'store', 'previous', 'previously', 'moreabout'
              ])

    printable = set(string.printable)
    text = hard_replaces(text)
    text = text.lower()
    text = text.encode("ascii", errors="ignore").decode() #remove non ascii
    text = ''.join([w for w in text if w in printable]).strip()

    tokenizer = RegexpTokenizer(r'((\w+-*)*\w+)')
    text = ' '.join([w[0] for w in tokenizer.tokenize(text)
        if not w[0] in stopwords 
        # and w.isalnum()
        and not w[0] in unfrequent
        ][:max_words])
    return text

def get_unfrequent(texts, cutoff=0):
    """
        Returns all tokens that appear in <= than `cutoff` texts
    """
    word_counts = collections.Counter()
    for phrase in texts:
        if not phrase:
            continue
        for word in set(phrase.split(' ')):
            word_counts[word] += 1

    return set([k for k, v in word_counts.items() if v <= cutoff])

def preprocess_text_column(texts, cutoff=0):
    if cutoff == 0:
        unfrequent = set()
    else:
        unfrequent = get_unfrequent(texts, cutoff=cutoff)
    new_texts = np.array(texts.apply(lambda x: preprocess_text_field(x, unfrequent)))
    return new_texts

def get_known_brands(df, freq_cutoff=5, blacklist=brand_blacklist):
    known_brands= get_known_items(df, 'brand', freq_cutoff, blacklist, additional_items=major_camera_brands)
    return known_brands

def preprocess_model(text):
    text = text.lower().strip()
    if len(text) < 3:
        return None
    text = re.sub(r'\W+', '', text)
    if text:
      text = re.search(special_token_pattern, text)
      if text:
        return text[0]
      else:
        return None

def get_known_models(df, freq_cutoff=1, blacklist=['digital']):
    known_models = get_known_items(df, 'model', freq_cutoff, blacklist, preprocessor=preprocess_model)
    return known_models

def preprocess_brand_field(specs_df, brand_blacklist, brand_cutoff=15):
    known_brands = get_known_brands(specs_df)

    populated_brands = np.array(populate_categories(specs_df.all_text.values, specs_df.brand.values, 'brand', set(known_brands), set(brand_blacklist) ))
    assert populated_brands.shape == specs_df['brand'].shape
    specs_df['brand'] = populated_brands

    populated_brands = np.array(populate_categories(specs_df.page_title.values, specs_df.brand.values, 'brand', set(known_brands), set(brand_blacklist) ))
    assert populated_brands.shape == specs_df['brand'].shape
    specs_df['brand'] = populated_brands

    # Make infrequent brands None using brand_cutoff
    brand_counts = specs_df['brand'].value_counts()
    cutoff_brand_counts = brand_counts[brand_cutoff:]
    cutoff_brands = list(cutoff_brand_counts.index)
    specs_df['brand'] = specs_df['brand'].apply(lambda brand: None if brand in cutoff_brands or not brand else brand)
    return specs_df

def populate_brands_from_models(specs_df):
    gdf = specs_df.groupby(['model', 'brand'])['spec_id'].nunique().reset_index().sort_values(by=['model', 'spec_id'])

    model_brands = {}
    for model in gdf.model.unique():
        row = gdf[gdf.model == model].iloc[0]
        model_brands[row.model] = row.brand

    def populate_brand(brand, model):
        if brand or not model or pd.isnull(model):
            return brand

        return model_brands.get(model)

    specs_df['brand'] = specs_df[['brand', 'model']].apply(lambda x: populate_brand(*x), axis=1)
    return specs_df

def preprocess_model_field(specs_df):
    known_models = get_known_models(specs_df)

    populated_models = np.array(populate_models(specs_df.all_text.values, specs_df.model.values, set(known_models), blacklist=[]))
    assert populated_models.shape == specs_df['model'].shape
    specs_df['model'] = populated_models

    populated_models = np.array(populate_models(specs_df.page_title.values, specs_df.model.values, set(known_models), blacklist=[]))
    assert populated_models.shape == specs_df['model'].shape
    specs_df['model'] = populated_models

    # Make infrequent models None 
    model_counts = specs_df['model'].value_counts()
    cutoff_model_counts = model_counts[model_counts < 2]
    cutoff_models = set(cutoff_model_counts.index)
    specs_df['model'] = specs_df['model'].apply(lambda model: None if model in cutoff_models or not model else model)
    return specs_df

def get_known_types(df, freq_cutoff=150, blacklist=[]):
    known_types= get_known_items(df, 'type', freq_cutoff, blacklist)
    return known_types

def preprocess_type_field(specs_df):
    blacklist = ['dig', 'lens']
    freq_cutoff = 150
    known_types = get_known_types(specs_df, freq_cutoff=freq_cutoff, blacklist=blacklist)

    populated_types = np.array(populate_categories(specs_df.all_text.values, specs_df.type.values,  'type', set(known_types), blacklist=blacklist))
    specs_df['type'] = populated_types

    # Make infrequent models None 
    type_counts = specs_df['type'].value_counts()
    cutoff_type_counts = type_counts[type_counts < freq_cutoff]
    cutoff_types = set(cutoff_type_counts.index)
    specs_df['type'] = specs_df['type'].apply(lambda type_: None if type_ in cutoff_types or not type_ else type_)

    return specs_df

def preprocess_megapixels_field(specs_df):
    known_items = get_known_items(specs_df, 'megapixels', 0, [])

    populated_mps = np.array(populate_categories(specs_df.all_text.values, specs_df.megapixels.values,  'megapixels', set(known_items), blacklist=[]))
    specs_df['megapixels'] = populated_mps

    return specs_df

def drop_rows(specs_df, drop_brands):
    # Drop specs with brands in drop_brands
    print('Dropping', specs_df[specs_df.brand.isin(drop_brands)].shape[0], 'known non-camera brand specs')
    specs_df = specs_df[~specs_df.brand.isin(drop_brands)]

    # Drop supposed camera bags and cases
    camera_types = ['dslr', 'underwater', 'point shoot', 'mirrorless', 'bridge']
    non_camera_index = ((specs_df.page_title_stem.str.contains('bag') | specs_df.page_title_stem.str.contains('case') | specs_df.page_title_stem.str.contains('backpack')) \
                        & ((specs_df.brand.isnull() & specs_df.model.isnull()) | (specs_df.type.isnull() & specs_df.brand.isnull()) | (specs_df.type.isnull() & specs_df.model.isnull())) \
                        & (~specs_df.type.isin(camera_types)) & specs_df.megapixels.isnull())

    print('Dropping', specs_df[non_camera_index].shape[0], 'camera bag specs')
    specs_df = specs_df[~non_camera_index]

    # Drop cctv cameras
    cctv_index = (specs_df.page_title.str.contains('cctv') 
              | specs_df.page_title.str.contains('surveillance') 
              | specs_df.page_title.str.contains('security') 
              | specs_df.page_title.str.contains(' ip ')
              | specs_df.page_title.str.contains('hikvis')
              | specs_df.page_title.str.contains('dahua')
              | specs_df.page_title.str.contains('onvif')
              | specs_df.page_title.str.contains('dome')) & specs_df.type.isnull()
    print('Dropping', specs_df[cctv_index].shape[0], 'cctv specs')
    specs_df = specs_df[~cctv_index]

    # Drop null titles
    bad_row_selector = (specs_df.page_title.isnull()) | (specs_df.page_title=='') | (specs_df.page_title=='null')
    null_rows = specs_df[bad_row_selector].shape[0]
    specs_df[bad_row_selector].all_text.values
    if null_rows:
        specs_df = specs_df[~bad_row_selector]
        print(f'Warning, dropped {null_rows} rows containing null page titles')
    return specs_df

def preprocess_specs_dataset(specs_df, 
                             max_words=500,
                             cutoff=5,
                             brand_blacklist=brand_blacklist,
                             brand_cutoff=15,
                             drop_brands=drop_brands,
                             ):
    """
        * lowercase
        * fix known common typos
        * remove nltk stopwords + some hand-picked stop words
        * only ascii symbols
        * only `string.printable` symbols
        * only alphanumeric symbols
        * max 500 words
        * remove words that appear only once
        * drop rows where page title is empty or null

        * populate brand and model fields from all_text
    """
    specs_df = specs_df.copy()
    brand_blacklist = brand_blacklist or []

    snow = nltk.stem.SnowballStemmer('english')
            
    def stem(text):
        return ' '.join([snow.stem(w) for w in text.split(' ')]).strip()

    # Clean up
    specs_df['page_title'] = preprocess_text_column(specs_df.page_title)

    specs_df['brand'] = preprocess_text_column(specs_df.brand)

    specs_df['all_text'] = preprocess_text_column(specs_df.all_text, cutoff=cutoff)

    # Stem
    specs_df['page_title_stem'] = specs_df.page_title.apply(stem)

    specs_df['all_text_stem'] = specs_df.all_text.apply(stem)

    # Site field
    
    specs_df['site'] = specs_df.spec_id.apply(extract_site)

    # Brand field

    specs_df = preprocess_brand_field(specs_df, brand_blacklist, brand_cutoff)

    # Model field

    specs_df = preprocess_model_field(specs_df)

    # Fill brands using models

    specs_df = populate_brands_from_models(specs_df)

    # Type field

    specs_df = preprocess_type_field(specs_df)

    # Megapixels field

    specs_df = preprocess_megapixels_field(specs_df)

    # Drop rows

    specs_df = drop_rows(specs_df, drop_brands)

    return specs_df
