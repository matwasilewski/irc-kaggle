import os.path
import pickle

from src.irc_kaggle.dataset import (
    original_greeks_df,
    original_test_df,
    original_train_df,
)
from src.irc_kaggle.preprocessing import make_preprocess_pipeline
from src.irc_kaggle.preprocessing_utils import (
    fix_columns_in_test_ds,
    get_r2_scores,
    get_times_from_greeks,
    get_transform_cols,
    scale_epsilon,
)
from src.irc_kaggle.training import (
    models_for_hyperparameter_optimization,
    tune_hyperparameters,
)


def greeks_pipeline():
    # With Greeks
    train_with_greeks_df = original_train_df.copy()
    train_with_greeks_df['Epsilon'] = get_times_from_greeks(original_greeks_df)

    # Test, with Greeks
    test_with_greeks_df = original_test_df.copy()
    test_with_greeks_df['Epsilon'] = train_with_greeks_df.Epsilon.max() + 1

    fix_columns_in_test_ds(test_with_greeks_df)
    train_with_greeks_transform_cols = get_transform_cols(
        get_r2_scores(train_with_greeks_df)
    )

    greeks_preprocess_pipeline = make_preprocess_pipeline(
        train_with_greeks_transform_cols
    )
    train_df = scale_epsilon(train_with_greeks_df)
    test_df = scale_epsilon(test_with_greeks_df)
    return greeks_preprocess_pipeline, train_df, test_df


def no_greeks_pipeline():
    train_no_greeks_df = original_train_df.copy()
    test_no_greeks_df = original_test_df

    fix_columns_in_test_ds(test_no_greeks_df)
    train_with_no_greeks_transform_cols = get_transform_cols(
        get_r2_scores(train_no_greeks_df)
    )

    no_greeks_preprocess_pipeline = make_preprocess_pipeline(
        train_with_no_greeks_transform_cols
    )

    return train_no_greeks_df, test_no_greeks_df, no_greeks_preprocess_pipeline


def tune_hyperparameters_on_greeks(artefact_dir_path):
    greeks_preprocess_pipeline, train_df, test_df = greeks_pipeline()
    models_params = models_for_hyperparameter_optimization()
    tuned_grid_searches = {}

    for model_args in models_params:
        print(f"Tuning hyper-parameters for model: {model_args['clf_name']}")
        classifier = model_args["clf"]
        classifier_name = model_args["clf_name"]
        param_grid = model_args["param_grid"]

        tuned_grid_searches[classifier_name] = tune_hyperparameters(
            train_df,
            greeks_preprocess_pipeline,
            classifier,
            classifier_name,
            param_grid,
        )

    for tuned_grid in tuned_grid_searches:
        pickle_name = f"{tuned_grid['clf_name']}_tuned_grid.pkl"

        with open(os.path.join(artefact_dir_path, pickle_name), 'wb') as f:
            pickle.dump(tuned_grid, f)
