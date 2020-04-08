import os
import re
import itertools
from tqdm import trange, tqdm
from pathlib import Path
import pandas as pd
import collections
from .preprocessing import preprocess_model
from .utils import (read_json, extract_json_text)


def type_postprocessor(item):
    if not item or len(item) < 3:
        return None

    item = item.lower().strip()
    item = ''.join([c for c in item if c.isalnum() or c == ' '])
    item = ' '.join([s for s in item.split(' ') if s])

    dslr_markers = ['digital', 'dslr', 'powershot', 'lcd display']
    for s in dslr_markers:
        if s in item:
            return 'dslr'

    slr_markers = ['slr']
    for s in slr_markers:
        if s in item:
            return 'slr'

    ps_markers = ['point shoot', 'point and shoot']
    for s in ps_markers:
        if s in item:
            return 'point shoot'

    mirrorless_markers = ['mirrorless']
    for s in mirrorless_markers:
        if s in item:
            return 'mirrorless'

    action_markers = ['action', 'sport']
    for s in action_markers:
        if s in item:
            return 'action'

    camcorder_markers = ['camcorder']
    for s in camcorder_markers:
        if s in item:
            return 'camcorder'

    bridge_markers = ['bridge']
    for s in bridge_markers:
        if s in item:
            return 'bridge'

    case_markers = ['case']
    for s in case_markers:
        if s in item:
            return 'case'

    return item

def megapixels_postprocessor(item):
    if not item or len(item) <= 1:
        return None

    matches = re.search(r'((\d+)[\s.]+){0,1}(\d+)[\s.]*((mp)|(megapixel))s?', item, flags=re.IGNORECASE)
    if matches:
        num1, num2 = matches[2], matches[3]
        if not num1 and not num2:
            return None
        if not num1:
            num1 = num2
            num2 = None
        if num2 == '0' or not num2:
            item = f'{num1}'
        else:
            item = f'{num1}-{num2}'
        item = item + '-mp'
    else:
        item = None
    return item


def parse_json_field_category(js, field_name, known_items, postprocessor=None, tokenizer=None):
    """Parse fields like brand, model, etc"""
    def default_postprocessor(item):
        if not item or len(item) < 3:
            return None
        item = re.sub(r'\W+', '', item)
        return item.lower().strip()
    postprocessor = postprocessor or default_postprocessor

    def default_tokenizer(item):
        return item.strip().split(' ')
    tokenizer = tokenizer or default_tokenizer

    item = js.get(field_name)

    # If there are multiple words, convert to list of words
    if isinstance(item, str) and len(tokenizer(item)) > 1:
        item = tokenizer(item)

    # Iterate over list, try to find a known item in it    
    if isinstance(item, list):
        for item in item:
            for known_item in known_items:
                if known_item in str(item).lower():
                    item = known_item
                    break
            if not isinstance(item, list):
                # Found
                break

    # If not found a known item perhaps its a new item
    if isinstance(item, list):
        try:
            item = str(item[0])
        except IndexError:
            print(f'Weird item in spec {js}')
    if item:
        item = postprocessor(item)
    return item


def make_specs_dataset(specs_path):
    site_folders = os.listdir(specs_path)
    known_brands = set()
    known_models = set()
    known_types = set()
    known_megapixels = set()
    Row = collections.namedtuple('Row', ['spec_id', 'page_title', 'brand', 'model', 'type', 'megapixels', 'all_text'])
    rows = []
    for site in site_folders:
        for fname in os.listdir(os.path.join(specs_path, site)):
            path = os.path.join(specs_path, site, fname)
            parsed = read_json(path)
            all_text = '\t'.join(extract_json_text(parsed))
            brand = parse_json_field_category(parsed, 'brand', known_brands)
            model = parse_json_field_category(parsed, 'model', known_models, postprocessor=preprocess_model)
            type_ = parse_json_field_category(parsed, 'type', known_types, tokenizer= lambda x: x, postprocessor=type_postprocessor)
            megapixels = parse_json_field_category(parsed, 'megapixels', known_megapixels, tokenizer= lambda x: x, postprocessor=megapixels_postprocessor)

            if isinstance(brand, str):
                known_brands.add(brand)

            if isinstance(model, str):
                known_models.add(model)

            if isinstance(type_, str):
                known_types.add(type_)

            if isinstance(megapixels, str):
                known_megapixels.add(megapixels)

            row = Row(site+'//'+fname.split('.')[0], 
                parsed['<page title>'], 
                brand,
                model,
                type_,
                megapixels,
                all_text)
            rows.append(row)
    specs_df = pd.DataFrame(rows)
    return specs_df