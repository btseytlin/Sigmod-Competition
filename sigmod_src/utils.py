import json
import os

major_camera_brands = ['vivitar', 'visiontek', 'vageeswari', 'traveler', 'thomson',
                       'tevion', 'sony', 'sigma', 'samsung', 'rollei', 'ricoh', 'praktica', 
                       'polaroid', 'phase one', 'pentax', 'panasonic', 'olympus', 'nikon',
                       'minox', 'memoto', 'medion', 'lytro', 'leica', 'kodak', 'hp',
                       'hasselblad', 'gopro', 'genius', 'fujifilm', 'foscam', 
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
