from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

from src.irc_kaggle.settings import ModelGreekSettings, ModelNoGreekSettings


def get_lgbm_model_greeks():
    lgbm_clf = LGBMClassifier(
        class_weight="balanced", **ModelGreekSettings.LGBM_PARAMS
    )
    return lgbm_clf


def get_xgb_model_greeks():
    xgb_clf = XGBClassifier(
        scale_pos_weight=4.71,
        random_state=ModelGreekSettings.SEED,
        **ModelGreekSettings.XGB_PARAMS,
    )
    return xgb_clf


def get_catboost_model_greeks():
    catboost_clf = CatBoostClassifier(
        auto_class_weights="Balanced", **ModelGreekSettings.CATBOOST_PARAMS
    )
    return catboost_clf


def get_models_greeks():
    return [
        ("CatBoost", get_catboost_model_greeks()),
        ("XGBoost", get_xgb_model_greeks()),
        ("LGBM", get_lgbm_model_greeks()),
    ]


def get_lgbm_model_no_greeks():
    lgbm_clf = LGBMClassifier(
        class_weight="balanced", **ModelNoGreekSettings.LGBM_PARAMS
    )
    return lgbm_clf


def get_xgb_model_no_greeks():
    xgb_clf = XGBClassifier(
        scale_pos_weight=4.71,
        random_state=ModelNoGreekSettings.SEED,
        **ModelNoGreekSettings.XGB_PARAMS,
    )
    return xgb_clf


def get_catboost_model_no_greeks():
    catboost_clf = CatBoostClassifier(
        auto_class_weights="Balanced", **ModelNoGreekSettings.CATBOOST_PARAMS
    )
    return catboost_clf


def get_models_no_greeks():
    return [
        ("CatBoost", get_catboost_model_no_greeks()),
        ("XGBoost", get_xgb_model_no_greeks()),
        ("LGBM", get_lgbm_model_no_greeks()),
    ]


def voting_classified(greeks: bool):
    if greeks is True:
        estimators = get_models_greeks()
    else:
        estimators = get_models_no_greeks()

    voting_clf = VotingClassifier(
        estimators=estimators,
    )
    return voting_clf
