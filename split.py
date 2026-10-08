"""
Part of Phase 7: train/validation/test split
------------------------------------------------
WHY A CHRONOLOGICAL (month-based) SPLIT: delay patterns have real
temporal structure in this data (see eda.py -- month-level seasonality
is substantial: 14.4% delay rate in September vs. 30.3% in July). A
random split would train and test on the same seasonal mix, which
overstates how well the model would actually perform when deployed
forward in time onto a DIFFERENT season than it trained on -- the same
principle behind the time-based splits in the sales-forecasting and
e-commerce-demand projects, applied here because this dataset, like
those, has genuine calendar structure a random split would hide.

WHY THESE SPECIFIC MONTH RANGES: train on the first 8 months (Jan-Aug),
validate on Sep-Oct (a genuinely different, lower-delay season than
training averages), test on Nov-Dec (which includes the December
holiday-travel spike) -- this means the test evaluation covers a
meaningfully different seasonal regime than most of training, a
harder and more honest test than a same-season random split would be.
"""

import pandas as pd

TRAIN_MONTHS = list(range(1, 9))    # Jan-Aug
VAL_MONTHS = [9, 10]                # Sep-Oct
TEST_MONTHS = [11, 12]              # Nov-Dec


def split_data(df: pd.DataFrame):
    train_df = df[df["month"].isin(TRAIN_MONTHS)].copy()
    val_df = df[df["month"].isin(VAL_MONTHS)].copy()
    test_df = df[df["month"].isin(TEST_MONTHS)].copy()
    return train_df, val_df, test_df
