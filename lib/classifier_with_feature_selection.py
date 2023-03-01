from sklearn.base import BaseEstimator
class ClassifierFeatureSelection(BaseEstimator):

    def __init__(self, classifier, feature_selector):
        self.classifier = classifier
        self.feature_selector = feature_selector

    def fit(self, X, y):
        # run the data through the feature selector
        X_r = self.feature_selector.fit_transform(X, y)
        self.classifier.fit(X_r, y)

    def predict(self, X):
        X_r = self.feature_selector.transform(X)
        y_pred = self.classifier.predict(X_r)
        return y_pred

    def predict_proba(self, X):
        X_r = self.feature_selector.transform(X)
        return self.classifier.predict_proba(X_r)