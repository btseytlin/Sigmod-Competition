import os
import sys
import random
import json
import collections
import re
from itertools import combinations

import pandas as pd
import numpy as np
import scipy
import statsmodels

from sigmod_src.make_dataset import make_specs_dataset
from sigmod_src.preprocessing import preprocess_specs_dataset
from sigmod_src.pipeline import LGBMPipeline
from sigmod_src.utils import get_additional_labels, make_classes_df

LG_LABELS_PATH = '../data/raw/sigmod_large_labelled_dataset.csv'
SPECS_PATH = '../data/raw/2013_camera_specs/'

labels_df = pd.read_csv(LG_LABELS_PATH)
specs_dataset_src = make_specs_dataset(SPECS_PATH)

specs_df = preprocess_specs_dataset(specs_dataset_src)
specs_df.index = specs_df.spec_id
specs_df.to_csv('../data/processed/specs.csv', index=None)

model = LGBMPipeline(specs_df, labels_df)
model.train()

model.make_submission()

print(f'Done! The submission file is at {model.submit_fpath}')