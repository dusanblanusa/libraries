import pandas as pd
import numpy as np
import shap


class ShapStats:
    def __init__(self, model, X, feature_names):
        """
        :param model: Fitted model
        :param X - feature values
        :feature names: list of feature names
        """
        self.model = model
        self.X = X
        if hasattr(X, "toarray"):
            X = X.toarray()

        X_df = pd.DataFrame(X, columns=feature_names)

        self.explainer = shap.Explainer(model, X_df)
        self.shap_values = self.explainer(X_df)

        try:
            explainer = shap.Explainer(model)
        except:
            explainer = shap.Explainer(model, X_df, feature_names=X_df.columns)

        self.shap_values = explainer(X_df)
    
    def get_feature_importance(self):
        mean_abs_shap = np.mean(np.abs(self.shap_values.values), axis=0)
        importance_df = pd.DataFrame({"Feature": self.X.columns, 
                                     "Importance":mean_abs_shap})
        return importance_df.sort_values(by='Importance', ascending=False)

    def global_bar(self, max_display=20):
        shap.plots.bar(self.shap_values, max_display=max_display)
        
    def beeswarm(self, max_display=20):
        shap.plots.beeswarm(self.shap_values, max_display=max_display)

    def scatter_local(self, feature):
        shap.plots.scatter(self.shap_values[:, feature])
  
        
def generate_shap_stats(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return shap_values
