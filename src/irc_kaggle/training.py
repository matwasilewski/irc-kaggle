from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.irc_kaggle.models import get_models_greeks, get_models_no_greeks
from src.irc_kaggle.settings import TuningSettings


def hyperparameter_grid_search(
    df, preprocessing_pipeline, classifier, classifier_name, param_grid
):
    assert classifier_name != "preprocessing"

    X = df.drop("Class", axis=1)
    y = df.Class

    param_grid = {f"{classifier_name}__{k}": v for k, v in param_grid.items()}

    fine_tuninng_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing_pipeline),
            (classifier_name, classifier),
        ]
    )
    grid_search = GridSearchCV(
        fine_tuninng_pipeline, param_grid, scoring="neg_log_loss"
    )
    grid_search.fit(X, y)

    print(f"Finished hyperparameter fine-tuning in {classifier_name}.")
    print(f"Best parameter (CV score={grid_search.best_score_}):")
    print(grid_search.best_params_)
    return grid_search


def xgb_model_for_optimization():
    xgb_clf = XGBClassifier(
        scale_pos_weight=4.71, random_state=TuningSettings.SEED
    )
    xgb_name = "XGBlassifier"
    xgb_arg = {
        "clf": xgb_clf,
        "clf_name": xgb_name,
        "param_grid": TuningSettings.XGB_GRID_PARAMS,
    }
    return xgb_arg


def lgbm_model_for_optimization():
    lgbm_clf = LGBMClassifier(class_weight="balanced")
    lgbm_name = "LGBMClassifier"
    lgbm_arg = {
        "clf": lgbm_clf,
        "clf_name": lgbm_name,
        "param_grid": TuningSettings.LGBM_GRID_PARAMS,
    }
    return lgbm_arg


def catboost_model_for_optimization():
    cat_clf = CatBoostClassifier(auto_class_weights="Balanced")
    cat_name = "CatBoostClassifier"
    cat_arg = {
        "clf": cat_clf,
        "clf_name": cat_name,
        "param_grid": TuningSettings.CATBOOST_GRID_PARAMS,
    }
    return cat_arg


def voting_model_for_optimization(greeks):
    if greeks:
        voting_clf = VotingClassifier(
            estimators=get_models_greeks(),
        )
    else:
        voting_clf = VotingClassifier(
            estimators=get_models_no_greeks(),
        )

    name = "VotingClassifier"
    arg = {
        "clf": voting_clf,
        "clf_name": name,
        "param_grid": TuningSettings.VOTING_GRID_PARAMS,
    }
    return arg


def models_for_hyperparameter_optimization():
    classifier_params = [
        lgbm_model_for_optimization(),
        xgb_model_for_optimization(),
        catboost_model_for_optimization(),
    ]
    return classifier_params


def models_for_ensamble_voting_optimization(greeks):
    classifier_params = [
        voting_model_for_optimization(greeks),
    ]
    return classifier_params
