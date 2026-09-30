"""Reusable data cleaning functions for the ImmoScout24 rental dataset.

Typical use:
    from cleaning import load_raw, clean_data, prepare_features
    df = clean_data(load_raw("data/immo_data.csv"))
    X, y = prepare_features(df), df[TARGET]
"""

import pandas as pd

TARGET = "baseRent"

# Columns that contain (part of) the answer. Using them would be "data leakage":
# the model would look great in testing but be useless for a new apartment.
LEAKAGE_COLUMNS = [
    "totalRent",        # baseRent + service charges -> contains the target
    "baseRentRange",    # the target, put into bins
    "serviceCharge",    # extra costs, strongly tied to total rent, not known to our app users
    "heatingCosts",
]

# Columns that are IDs, free text, duplicates of other columns or unrelated to the flat.
UNUSED_COLUMNS = [
    "scoutId", "description", "facilities",                     # ID and free text
    "street", "streetPlain", "houseNumber",                     # too detailed, many missing
    "geo_bln", "geo_krs",                                       # same info as regio1 / regio2
    "livingSpaceRange", "noRoomsRange", "yearConstructedRange",  # binned copies of other columns
    "telekomTvOffer", "telekomUploadSpeed", "telekomHybridUploadSpeed",  # internet offers
    "picturecount", "pricetrend", "date",                       # about the listing, not the flat
]

# Realistic value ranges (anything outside is most likely a typo or a fake listing).
RENT_RANGE = (100, 5000)           # euros per month
SPACE_RANGE = (10, 300)            # square metres
ROOMS_RANGE = (1, 10)
YEAR_RANGE = (1800, 2026)

# Columns used to recognise the same apartment listed several times (e.g. in different months).
DUPLICATE_KEY = ["regio2", "regio3", "street", "livingSpace", "noRooms",
                 "baseRent", "yearConstructed", "floor"]

# Features the final model uses (everything a user can type into the app).
NUMERIC_FEATURES = ["livingSpace", "noRooms", "yearConstructed"]
BOOLEAN_FEATURES = ["balcony", "hasKitchen", "lift", "cellar", "garden", "newlyConst"]
CATEGORICAL_FEATURES = ["regio1", "regio2", "condition", "typeOfFlat", "interiorQual"]
FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def load_raw(path):
    """Read the raw Kaggle CSV."""
    return pd.read_csv(path)


def drop_duplicate_listings(df):
    """Keep only one copy of apartments that were listed more than once."""
    return df.drop_duplicates(subset=DUPLICATE_KEY)


def drop_mostly_empty_columns(df, threshold=0.5):
    """Drop columns where more than `threshold` (e.g. 50%) of the values are missing."""
    missing_share = df.isna().mean()
    return df.loc[:, missing_share <= threshold]


def drop_unused_columns(df):
    """Drop leakage columns and columns we don't need (IDs, text, duplicates)."""
    to_drop = [c for c in LEAKAGE_COLUMNS + UNUSED_COLUMNS if c in df.columns]
    return df.drop(columns=to_drop)


def filter_realistic_values(df):
    """Remove rows with unrealistic rent, size, rooms or construction year.

    A missing construction year is kept: missing is not the same as wrong,
    and the model can handle it.
    """
    year_ok = df["yearConstructed"].between(*YEAR_RANGE) | df["yearConstructed"].isna()
    mask = (
        df[TARGET].between(*RENT_RANGE)
        & df["livingSpace"].between(*SPACE_RANGE)
        & df["noRooms"].between(*ROOMS_RANGE)
        & year_ok
    )
    return df.loc[mask]


def clean_data(df):
    """Run all cleaning steps in order and return a fresh, re-indexed DataFrame."""
    df = drop_duplicate_listings(df)
    df = drop_mostly_empty_columns(df)
    df = drop_unused_columns(df)
    df = filter_realistic_values(df)
    return df.reset_index(drop=True)


def prepare_features(df):
    """Select the model features and give them clean, model-friendly types.

    - Booleans become 0/1.
    - Missing categories become the explicit category "unknown".
    - Text columns become pandas 'category' type (LightGBM uses this directly).
    """
    X = df[FEATURES].copy()
    X[BOOLEAN_FEATURES] = X[BOOLEAN_FEATURES].astype(int)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("unknown").astype("category")
    return X
