import numpy as np
from sklearn.feature_selection import f_classif
from scipy.stats import f_oneway, kruskal, permutation_test
from multitest import MultiTest
import warnings

def groups_f_stat(*groups, axis=0):
    f_stat, pval = f_oneway(*groups,axis=axis)
    # replace NaNs in the F statistic with zeros
    f_stat[np.isnan(f_stat)] = 0
    return f_stat

class FeatureSelectionBase(object):

    def __init__(self, verbosity=False):
        self.selected_features = None
        self.verbosity = verbosity

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
        self.mask = None
        self.hc = None
        self.hct = 1
        self.num_features = None

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
        self.num_features = X.shape[1]
        # init an array for the pvalues
        self.pvals = np.ones((self.num_features,))
        self.f_stat = np.zeros_like(self.pvals)
        self.mask = np.zeros_like(self.pvals)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_for_test_f_stat, features_for_test_pvals, inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx = self.test_func(samples)
        # where the f_stat is infinite set the pval to 0
        self.pvals[inf_f_stat_idx] = 0
        self.f_stat[inf_f_stat_idx] = np.inf
        self.mask[inf_f_stat_idx] = 1
        # where the f_stat is nan set the pval to 1 - degenerated features
        self.pvals[nan_f_stat_idx] = 1
        self.f_stat[nan_f_stat_idx] = 0
        # apply higher criticism on the rest of the features
        self.pvals[features_for_test_idx] = features_for_test_pvals
        self.f_stat[features_for_test_idx] = features_for_test_f_stat
        mask = self.apply_hc(features_for_test_pvals)
        hc_selected_features_idx = features_for_test_idx[np.flatnonzero(mask)]
        # update the feature selector mask with the hc features
        self.mask[hc_selected_features_idx] = 1
        self.selected_features = np.flatnonzero(self.mask)

    def get_features_for_test(self, groups):
        # first, compute the statistic for all features
        all_features_f_stat, _ = f_oneway(*groups, axis=0)
        # find where the f_stat is infinite - these are features we'll definitely want to take, since their "within"
        # variance is 0
        inf_f_stat_idx = np.isinf(all_features_f_stat)
        # find where the f_stat is undefined - these are degenerated features
        nan_f_stat_idx = np.isnan(all_features_f_stat)
        if self.verbosity:
            print(f"In feature selector {self.__class__.__name__}, number of inf features = {np.sum(inf_f_stat_idx)}")
            print(f"In feature selector {self.__class__.__name__}, number of nan features = {np.sum(nan_f_stat_idx)}")
        # get indices for the rest of the features
        features_for_test_idx = np.array(list(set(range(self.num_features)) - set(np.flatnonzero(inf_f_stat_idx)) - set(np.flatnonzero(nan_f_stat_idx))))
        #features_for_test_idx = np.array(list(set(range(self.num_features)) - set(np.flatnonzero(nan_f_stat_idx))))
        return inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx

    def test_func(self,groups):
        inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx = self.get_features_for_test(groups)
        if self.verbosity:
            print(f"In feature selector {self.__class__.__name__}, number of features for hc test = {len(features_for_test_idx)}")
        groups_for_test = [group[:, features_for_test_idx] for group in groups]
        features_for_test_f_stat, features_for_test_pvals = f_oneway(*groups_for_test)
        return features_for_test_f_stat, features_for_test_pvals, inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx


class FeatureSelectionDiversityPursuitPermutation(FeatureSelectionDiversityPursuitAnova):

    def __init__(self, n_resamples, hc_gamma=0.2):
        super().__init__(hc_gamma=hc_gamma)
        self.n_resamples = n_resamples
        self.null_dist = {}
    def test_func(self,groups):
        inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx = self.get_features_for_test(groups)
        groups_for_test = [group[:,features_for_test_idx] for group in groups]
        res = permutation_test(groups_for_test, groups_f_stat, n_resamples=self.n_resamples, vectorized=True, alternative='greater', permutation_type='independent')
        features_for_test_f_stat, features_for_test_pvals = res.statistic, res.pvalue
        return features_for_test_f_stat, features_for_test_pvals, inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx




class FeatureSelectionDiversityPursuitKruskal(FeatureSelectionDiversityPursuitAnova):

    def test_func(self,groups):
        num_features = groups[0].shape[1]
        # init arrays for f_stat, pvals
        f_stat = np.zeros((num_features,))
        pvals = np.ones((num_features,))
        for j in range(num_features):
            samples = [np.squeeze(group[:,j]) for group in groups]
            try:
                f_stat[j], pvals[j] = kruskal(*samples)
            except ValueError:
                # when all data is identical, the kruskal function will raise a value error. meaning for us that the feature is uninformative.
                # that is equivalent to the ANOVA giving nan value
                f_stat[j] = 0
                pvals[j] = 1
        # The kruskal H statistic cannot be infinite when at least some of the observations are different.
        inf_f_stat_idx = np.zeros((num_features,), dtype=bool)
        # get the indice of the "nan" H statistic
        nan_f_stat_idx = np.flatnonzero((pvals == 1))
        features_for_test_idx = np.array(list(set(range(self.num_features)) - set(np.flatnonzero(inf_f_stat_idx)) - set(np.flatnonzero(nan_f_stat_idx))))
        features_for_test_f_stat = f_stat[features_for_test_idx]
        features_for_test_pvals = pvals[features_for_test_idx]
        return features_for_test_f_stat, features_for_test_pvals, inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx

class FeatureSelectionOneVsAllAnova(FeatureSelectionDiversityPursuitAnova):

    def fit(self,X,y):
        # init the feature mask to all zeros (i.e. no features are selected)
        self.mask = np.zeros((X.shape[1],))
        self.num_features = X.shape[1]
        # collect all the stat/p-values in a dict
        self.f_stat = {t : np.ones_like(self.mask) for t in list(np.unique(y))}
        self.pvals = {t : np.ones_like(self.mask) for t in list(np.unique(y))}
        # go over all the targets
        for t in list(np.unique(y)):
            # get all samples with the current target as one group
            g1 = X[y==t,:]
            # get the rest of the samples (corresponding to all other targets) as the second group
            g2 = X[y!=t,:]
            # perform an ANOVA test between the two groups
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features_for_test_f_stat, features_for_test_pvals, inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx = self.test_func([g1, g2])
            # where the f_stat is infinite set the pval to 0
            self.pvals[t][inf_f_stat_idx] = 0
            self.f_stat[t][inf_f_stat_idx] = np.inf
            self.mask[inf_f_stat_idx] = 1
            # where the f_stat is nan set the pval to 1 - degenerated features
            self.pvals[t][nan_f_stat_idx] = 1
            self.f_stat[t][nan_f_stat_idx] = 0
            # apply higher criticism on the rest of the features
            self.pvals[t][features_for_test_idx] = features_for_test_pvals
            mask = self.apply_hc(features_for_test_pvals)
            hc_selected_features_idx = features_for_test_idx[np.flatnonzero(mask)]
            # update the feature selector mask with the hc features
            self.mask[hc_selected_features_idx] = 1


        # finally, get the features indices from the mask
        self.selected_features = np.flatnonzero(self.mask)