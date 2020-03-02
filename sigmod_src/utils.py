import json
import os

major_camera_brands = ['vivitar', 'visiontek', 'vageeswari', 'traveler', 'thomson',
                       'tevion', 'sony', 'sigma', 'samsung', 'rollei', 'ricoh', 'praktica', 
                       'polaroid', 'phase one', 'pentax', 'panasonic', 'olympus', 'nikon',
                       'minox', 'memoto', 'medion', 'lytro', 'leica', 'kodak', 'hp',
                       'hasselblad', 'gopro', 'genius', 'ge', 'fujifilm', 'foscam', 
                       'epson', 'casio', 'canon', 'blackmagic design',
                       'benq', 'bell & howell', 'aigo', 'agfaphoto', 'advert tech',
                       
                       'dahua', 'philips', 'fuji', 'sanyo', 'vizio', 'sharp',
                       'logitech', 'hikvision', 'bell', 'topixo', 'magnavox'
                      ]

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

def extract_brand(text):
    for brand in major_camera_brands:
        if brand in text.lower():
            return brand
    return None

def extract_site(spec_id):
    return spec_id.split('//')[0]

def get_vector_for_spec_id(spec_id, specs_df, spec_vectors):
    return spec_vectors[specs_df[specs_df.spec_id == spec_id].index][0]