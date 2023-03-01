import numpy as np
from sklearn.feature_selection import f_classif
from scipy.stats import f_oneway
from multitest import MultiTest
import warnings

class FeatureSelectionBase(object):

    def __init__(self):
        self.selected_features = None


    def fit(self,X,y):
        """
        Args:
        :param X:  data with shape (n_samples, n_features)
        :param y:  target values with shape (n_samples,)
        Returns:
        self: object
        """
        self.selected_features = np.arange(0,X.shape[1])

    def transform(self, X):
        """
        Reduce X to the selected features.
        Args:
        :param X:  data with shape (n_samples, n_features)
        Returns:
        X_r: data with shape [n_samples, n_selected_features]
        """
        if self.selected_features is None:
            return X
        else:
            return X[:,self.selected_features]

    def fit_transform(self, X, y=None):
        if self.selected_features is None:
            self.fit(X,y)
        # transform the data
        return self.transform(X)


class FeatureSelectionDiversityPursuitAnova(FeatureSelectionBase):

    def __init__(self, hc_gamma=0.2):
        super().__init__()
        self.test = f_oneway
        self.hc_gamma = hc_gamma
        self.f_stat = None
        self.pvals = None
        self.hc = None
        self.hct = 1

    def apply_hc(self,pvalues):
        """
        Applies higher criticism thresholding on the uni-variate features P-values
        Returns:
        self : object
        """
        mt = MultiTest(pvalues)
        self.hc, self.hct = mt.hc_star(gamma=self.hc_gamma)
        mask = (pvalues < self.hct)
        return mask


    def fit(self,X,y):
        samples = [X[y==label,:] for label in list(np.unique(y))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.f_stat, self.pvals = self.test_func(samples)
        self.mask = self.apply_hc(self.pvals)
        self.selected_features = np.flatnonzero(self.mask)

    def test_func(self,groups):
        f_stat, pvals = self.test(*groups)
        # find where the f_stat is infinite and set the pval to 0
        pvals[np.isinf(f_stat)] = 0
        # find where the f_stat is undefined and set the pval to 1
        pvals[np.isnan(f_stat)] = 1
        return f_stat, pvals


class FeatureSelectionOneVsAllAnova(FeatureSelectionDiversityPursuitAnova):

    def fit(self,X,y):
        # init the feature mask to all zeros (i.e. no features are selected)
        self.mask = np.zeros((X.shape[1],))
        # collect all the p-values in a dict
        self.pvals = {}
        # go over all the targets
        for t in list(np.unique(y)):
            # get all samples with the current target as one group
            g1 = X[y==t,:]
            # get the rest of the samples (corresponding to all other targets) as the second group
            g2 = X[y!=t,:]
            # perform an ANOVA test between the two groups
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f, pvalues = self.test_func([g1, g2])
            mask = self.apply_hc(pvalues)
            self.pvals[t] = pvalues
            # update the mask with the features obtained for the current one-vs-all anova test
            self.mask = np.logical_or(self.mask, mask)

        # finally, get the features indices from the mask
        self.selected_features = np.flatnonzero(self.mask)