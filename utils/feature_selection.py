from sklearn.feature_selection import f_classif, RFE
import xgboost as xgb
import pandas as pd
import numpy as np


def get_feature_importance_from_model(model, features):
    """
    Extract feature importances from provided model
    :param model: trained sklearn model
    :param features: list - names of the features
    """
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({"Feature": features, "importance": feature_importance})
    feature_importance_df.sort_values('importance', ascending=False)
    return feature_importance_df


def get_sorted_feature_importance_lr(model, feature_names: list):
    """
    Extraxt sorted features by their importances from logistic regression model
    """

    coef = model.coef_[0]
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": coef
    })
    
    importance_df["Absolute Importance"] = np.abs(importance_df["Importance"])
    importance_df = importance_df.sort_values(by="Absolute Importance", ascending=False)
    
    return importance_df[["Feature", "Importance"]]


def feature_selection_anova(X, y):
    """
    Feature selection by anova f value algorithm
    """
    f_value = f_classif(X, y)
    zipped = zip(X.columns, f_value[0])
    zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    return zipped


def feature_selection_rfe(X, y):
    """
    Feature selection by random feature elimination, using XGBoost as a model
    """
    rfe = RFE(xgb.XGBClassifier(), n_features_to_select=1)
    rfe.fit(X, y)
    ranking = rfe.ranking_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Ranking': ranking})
    feature_importance_df = feature_importance_df.sort_values(by='Ranking')
    return feature_importance_df
