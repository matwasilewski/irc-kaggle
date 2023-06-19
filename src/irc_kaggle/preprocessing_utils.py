from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import probplot
import scipy.stats as stats

from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


def scale_epsilon(df):
    if "Epsilon" not in df.columns:
        return df

    scaling_pipeline = make_pipeline(
        make_column_transformer(
            (
                MinMaxScaler((1, df.Epsilon.max() - df.Epsilon.min() + 2)),
                make_column_selector("Epsilon"),
            ),
            remainder="passthrough",
            verbose_feature_names_out=False,
        ),
    )
    df_scaled = pd.DataFrame(
        scaling_pipeline.fit_transform(df),
        columns=scaling_pipeline.get_feature_names_out(),
        index=df.index,
    )
    df_scaled = df_scaled.astype(df.dtypes)
    return df_scaled


def get_r2_scores(df):
    numeric_columns = (
        df.select_dtypes("number").drop("Class", axis=1).columns.tolist()
    )
    df = scale_epsilon(df)
    r2_scores = defaultdict(tuple)

    for feature in numeric_columns:
        orig = df[feature].dropna()
        _, (*_, R_orig) = probplot(orig, rvalue=True)
        _, (*_, R_log) = probplot(np.log(orig), rvalue=True)
        _, (*_, R_sqrt) = probplot(np.sqrt(orig), rvalue=True)
        _, (*_, R_reci) = probplot(np.reciprocal(orig), rvalue=True)
        _, (*_, R_boxcox) = probplot(stats.boxcox(orig)[0], rvalue=True)
        _, (*_, R_yeojohn) = probplot(stats.yeojohnson(orig)[0], rvalue=True)
        r2_scores[feature] = (
            R_orig * R_orig,
            R_log * R_log,
            R_sqrt * R_sqrt,
            R_reci * R_reci,
            R_boxcox * R_boxcox,
            R_yeojohn * R_yeojohn,
        )

    r2_scores = pd.DataFrame(
        r2_scores,
        index=(
            "Original",
            "Log",
            "Sqrt",
            "Reciprocal",
            "BoxCox",
            "YeoJohnson",
        ),
    ).T
    r2_scores["Winner"] = r2_scores.idxmax(axis=1)
    return r2_scores


def get_transform_cols(r2_scores):
    no_transform_cols = r2_scores.query("Winner == 'Original'").index
    log_transform_cols = r2_scores.query("Winner == 'Log'").index
    sqrt_transform_cols = r2_scores.query("Winner == 'Sqrt'").index
    reciprocal_transform_cols = r2_scores.query("Winner == 'Reciprocal'").index
    boxcox_transform_cols = r2_scores.query("Winner == 'BoxCox'").index
    yeojohnson_transform_cols = r2_scores.query("Winner == 'YeoJohnson'").index

    transform_cols = {
        "no_transform": no_transform_cols,
        "log_transform": log_transform_cols,
        "sqrt_transform": sqrt_transform_cols,
        "reciprocal_transform": reciprocal_transform_cols,
        "boxcox_transform": boxcox_transform_cols,
        "yeojohnson_transform": yeojohnson_transform_cols,
    }

    return transform_cols


def get_times_from_greeks(greeks_df):
    def convert_to_ordinal(date_str):
        if date_str != 'Unknown':
            return datetime.strptime(date_str, '%m/%d/%Y').toordinal()
        else:
            return np.nan

    times = greeks_df.Epsilon.map(convert_to_ordinal)
    return times


# Box-Cox requires that values in any column are not identical.
# This is extremely unlikely to be a case for any of the actual test cases, but is the case for the dummy test data provided in the dataset.
def fix_columns_in_test_ds(df):
    float_rows = df.dtypes == float

    for column in df.columns:
        values = df[column].values
        if np.all(values == values[0]) and df[column].dtype == float:
            df.loc[0, float_rows] += 1
