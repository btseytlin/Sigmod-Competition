import json
import itertools
import os
import re
import collections
import pandas as pd
import sys
from tqdm import tqdm

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
                       'colorpix', 'onvif', 'sjcam'
                      ]

brand_blacklist = ['shoot', 'as', 
            'eos', 'action', 'new', 'class', 'sharp', 'digital', 'sj4000']

special_token_pattern = re.compile(r'\b(([a-z][-\w]*[0-9])|([0-9][-\w]*[a-z])|([a-z][-\w]*[0-9][-\w]*[a-z]))\b')

number_token_pattern = re.compile(r'\b([0-9]+)\b')

def extract_special_tokens(string):
    # Tokens that resemble characteristics like model, camera lens param etc
    matches = [x[0] for x in re.findall(special_token_pattern, string)]
    return matches

def extract_number_tokens(string):
    # Tokens of only numbers
    matches = [x[0] for x in re.findall(number_token_pattern, string)]
    return matches

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

def get_known_brands(df, freq_cutoff=5, blacklist=brand_blacklist):
    known_brands= get_known_items(df, 'brand', freq_cutoff, blacklist, additional_items=major_camera_brands)
    return known_brands


def get_known_models(df, freq_cutoff=1, blacklist=[]):
    known_models = get_known_items(df, 'model', freq_cutoff, blacklist, preprocessor=preprocess_model)
    return known_models

def get_known_items(df, field_name, freq_cutoff, blacklist, additional_items=None, preprocessor=None):
    additional_items = []
    known_items = list(df[field_name].unique()) + additional_items
    if preprocessor:
      known_items = [preprocessor(x) for x in known_items if x and preprocessor(x)]
    known_items = set(known_items)
    known_items = list(known_items.difference(set(blacklist)))

    c = collections.Counter()
    for item in known_items:
        c[item] = df[df[field_name] == item].shape[0]

    item_counts = pd.Series(c).sort_values(ascending=False)
    known_items = item_counts[item_counts > freq_cutoff].index.tolist()
    known_items = [x.strip() for x in known_items if x.strip()]
    return known_items



def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def pprint_json(json_):
    print(json.dumps(json_, indent=4, sort_keys=True))

def path_from_spec_id(spec_id, prefix):
    if not prefix.endswith('/'):
        prefix = prefix + '/'
    site, fname = spec_id.split('//')
    return prefix + site + '/' + fname + '.json'

def extract_site(spec_id):
    return spec_id.split('//')[0]

def get_vector_for_spec_id(spec_id, specs_df, spec_vectors):
    return spec_vectors[specs_df[specs_df.spec_id == spec_id].index][0]

def extract_json_text(obj):
    results = []
    def _extract(obj, strings=None):
        strings = strings or []
        if isinstance(obj, str):
            strings.append(obj)
            return strings
        elif isinstance(obj, dict):
            for key in obj.keys():
                addition = _extract(obj[key])
                strings += addition
            return strings
        elif isinstance(obj, list):
            for item in obj:
                strings += _extract(item)
                return strings
        else:
            strings.append(str(obj))
            return strings
        
    results = _extract(obj, [])
    return results

def join_labels_specs(labels_df, specs_df):
    sides = ['left', 'right']
    for side in sides:
        join_df = specs_df.copy()
        join_df.columns = [side+'_'+col for col in join_df.columns]
        labels_df = labels_df.merge(join_df, on=side+'_spec_id', how='left')
    return labels_df


def get_additional_labels(labels_df, specs_df):
    class_ratio  = labels_df[labels_df.label==1].shape[0] / labels_df[labels_df.label==0].shape[0]

    labelled_spec_ids = set(list(labels_df.left_spec_id.values) + list(labels_df.right_spec_id.values))

    grouped = specs_df.groupby(['brand', 'model'])['spec_id'].agg(list).reset_index()

    grouped = grouped[~grouped.brand.isnull() & ~grouped.model.isnull() & grouped.spec_id.apply(lambda al: len(al) > 1)]

    new_dups = []
    for ix, row in tqdm(grouped.iterrows()):
        for comb in itertools.combinations(row.spec_id, 2):
            if comb[0] not in labelled_spec_ids and comb[1] not in labelled_spec_ids:
                new_dups.append((comb[0], comb[1], 1))

    new_non_dups = []
    for brand in tqdm(grouped.brand.unique()):
        brand_df = grouped[grouped.brand==brand]
        brand_models = brand_df.model.unique()
        combs = itertools.combinations(brand_models, 2)
        if combs:
            for model_pairs in combs:
                first_model_specs = brand_df[brand_df.model==model_pairs[0]].spec_id.values[0]
                second_model_specs = brand_df[brand_df.model==model_pairs[1]].spec_id.values[0]

                for left_spec_id in first_model_specs:
                    for right_spec_id in second_model_specs:
                        new_non_dups.append((left_spec_id, right_spec_id, 0))


    new_dups = pd.DataFrame(new_dups, columns=['left_spec_id', 'right_spec_id', 'label'])                 
    new_non_dups = pd.DataFrame(new_non_dups, columns=['left_spec_id', 'right_spec_id', 'label'])

    new_non_dups = new_non_dups.sample(int(new_dups.shape[0]/class_ratio)) # preserve class ratio

    return pd.concat([new_dups, new_non_dups], axis=0, ignore_index=True)

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