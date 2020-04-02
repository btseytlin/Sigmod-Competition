import os
import re
import itertools
from tqdm import trange, tqdm
from pathlib import Path
import pandas as pd
import collections
from .preprocessing import preprocess_model
from .utils import (read_json, extract_json_text)


def parse_json_field_category(js, field_name, known_items, postprocessor=None):
    """Parse fields like brand, model, etc"""
    def default_postprocessor(item):
        if not item or len(item) < 3:
            return None
        item = re.sub(r'\W+', '', item)
        return item.lower().strip()

    postprocessor = postprocessor or default_postprocessor

    item = js.get(field_name)

    # If there are multiple words, convert to list of words
    if isinstance(item, str) and len(item.strip().split(' ')) > 1:
        item = item.strip().split(' ')

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
    Row = collections.namedtuple('Row', ['spec_id', 'page_title', 'brand', 'model', 'all_text'])
    rows = []
    for site in site_folders:
        for fname in os.listdir(os.path.join(specs_path, site)):
            path = os.path.join(specs_path, site, fname)
            parsed = read_json(path)
            all_text = '\t'.join(extract_json_text(parsed))
            brand = parse_json_field_category(parsed, 'brand', known_brands)
            model = parse_json_field_category(parsed, 'model', known_models, postprocessor=preprocess_model)

            if isinstance(brand, str):
                known_brands.add(brand)

            if isinstance(model, str):
                known_models.add(model)

            row = Row(site+'//'+fname.split('.')[0], 
                parsed['<page title>'], 
                brand,
                model,
                all_text)
            rows.append(row)
    specs_df = pd.DataFrame(rows)
    return specs_df