import os.path

import pandas as pd

from src.irc_kaggle.preprocessing_utils import (
    fix_columns_in_test_ds,
    get_times_from_greeks,
)

dataset_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"
)

original_train_df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
original_test_df = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
original_greeks_df = pd.read_csv(os.path.join(dataset_dir, 'greeks.csv'))
original_sample_submission_df = pd.read_csv(
    os.path.join(dataset_dir, 'sample_submission.csv')
)

# Add 'Class' to Greeks
original_greeks_df = original_greeks_df.join(original_train_df.Class)
