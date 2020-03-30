import json
import os
import re
import collections
import pandas as pd
import sys

major_camera_brands = ['vivitar', 'visiontek', 'vageeswari', 'traveler', 'thomson',
                       'tevion', 'sony', 'sigma', 'samsung', 'rollei', 'ricoh', 'praktica', 
                       'polaroid', 'phase one', 'pentax', 'panasonic', 'olympus', 'nikon',
                       'minox', 'memoto', 'medion', 'lytro', 'leica', 'kodak', 'hp',
                       'hasselblad', 'gopro', 'genius', 'fujifilm', 'foscam', 
                       'epson', 'casio', 'canon', 'blackmagic design',
                       'benq', 'bell & howell', 'aigo', 'agfaphoto', 'advert tech',
                       
                       'dahua', 'philips', 'sanyo', 'vizio', 'sharp',
                       'logitech', 'hikvision', 'bell', 'topixo', 'magnavox',

                       'samyang', 'sekonic', 'lexar', 'ksm', 'uv', 'hoya', 'dahua',
                       'colorpix', 'onvif'
                      ]
def get_known_brands(df, freq_cutoff, blacklist):
    known_brands = list(df.brand.unique()) + major_camera_brands
    known_brands = set(known_brands)
    known_brands = list(known_brands.difference(set(blacklist)))

    c = collections.Counter()
    for brand in known_brands:
        c[brand] = df[df.brand == brand].shape[0]

    brand_counts = pd.Series(c).sort_values(ascending=False)
    known_brands = brand_counts[brand_counts > freq_cutoff].index.tolist()
    return known_brands


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

def extract_text(obj):
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

