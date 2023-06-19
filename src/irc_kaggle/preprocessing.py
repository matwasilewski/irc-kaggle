import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer,
    PowerTransformer,
    OrdinalEncoder,
)


def make_preprocess_pipeline(transform_cols):
    no_transform_cols = transform_cols["no_transform"]
    log_transform_cols = transform_cols["log_transform"]
    sqrt_transform_cols = transform_cols["sqrt_transform"]
    reciprocal_transform_cols = transform_cols["reciprocal_transform"]
    boxcox_transform_cols = transform_cols["boxcox_transform"]
    yeojohnson_transform_cols = transform_cols["yeojohnson_transform"]

    preprocess_pipeline = make_pipeline(
        make_column_transformer(
            (
                StandardScaler(),
                no_transform_cols.to_list(),
            ),
            (
                make_pipeline(
                    FunctionTransformer(
                        func=np.log, feature_names_out="one-to-one"
                    ),
                    StandardScaler(),
                ),
                log_transform_cols.to_list(),
            ),
            (
                make_pipeline(
                    FunctionTransformer(
                        func=np.log, feature_names_out="one-to-one"
                    ),
                    StandardScaler(),
                ),
                sqrt_transform_cols.to_list(),
            ),
            (
                make_pipeline(
                    FunctionTransformer(
                        func=np.reciprocal, feature_names_out="one-to-one"
                    ),
                    StandardScaler(),
                ),
                reciprocal_transform_cols.to_list(),
            ),
            (
                PowerTransformer(method="box-cox", standardize=True),
                boxcox_transform_cols.to_list(),
            ),
            (
                PowerTransformer(method="yeo-johnson", standardize=True),
                yeojohnson_transform_cols.to_list(),
            ),
            (
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
                make_column_selector(dtype_include=object),  # type: ignore
            ),
            remainder="passthrough",
            verbose_feature_names_out=False,
        ),
        KNNImputer(n_neighbors=10, weights="distance"),
    )
    return preprocess_pipeline
