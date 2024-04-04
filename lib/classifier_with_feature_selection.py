from sklearn.base import BaseEstimator
import numpy as np
class ClassifierFeatureSelection(BaseEstimator):

    def __init__(self, classifier, feature_selector):
        self.classifier = classifier
        self.feature_selector = feature_selector
        self.predict_random = False
        self.classes = None
        self.num_selected_features = None

    def fit(self, X, y):
        # run the data through the feature selector
        X_r = self.feature_selector.fit_transform(X, y)
        self.num_selected_features = X_r.shape[1]
        self.classifier.fit(X_r, y)

    def predict(self, X):
        X_r = self.feature_selector.transform(X)
        y_pred = self.classifier.predict(X_r)
        return y_pred

    def predict_proba(self, X):
        X_r = self.feature_selector.transform(X)
        pp = self.classifier.predict_proba(X_r)
        return pp