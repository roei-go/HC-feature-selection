from sklearn.base import BaseEstimator
import numpy as np
class ClassifierFeatureSelection(BaseEstimator):

    def __init__(self, classifier, feature_selector):
        self.classifier = classifier
        self.feature_selector = feature_selector
        self.predict_random = False
        self.classes = None

    def fit(self, X, y):
        # run the data through the feature selector
        X_r = self.feature_selector.fit_transform(X, y)
        if len(self.feature_selector.selected_features) == 0:
            print(f"In classifier with feature selector {self.feature_selector.__class__.__name__} no features were selected and classifier will not be fitted to data - will predict at random!")
            self.predict_random = True
            self.classes = list(np.unique(y))
        else:
            self.classifier.fit(X_r, y)

    def predict(self, X):
        if self.predict_random:
            y_pred = np.random.choice(self.classes, size=X.shape[0])
        else:
            X_r = self.feature_selector.transform(X)
            y_pred = self.classifier.predict(X_r)
        return y_pred

    def predict_proba(self, X):
        if self.predict_random:
            # return a uniform distribution over the classes
            pp = np.ones((X.shape[0],len(self.classes))) * (1/len(self.classes))
        else:
            X_r = self.feature_selector.transform(X)
            pp = self.classifier.predict_proba(X_r)
        return pp