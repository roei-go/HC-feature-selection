import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from scipy.stats import f_oneway, kruskal, permutation_test, anderson_ksamp, ks_2samp
from multitest import MultiTest
import warnings
from scipy.signal import find_peaks_cwt, peak_prominences
from scipy.stats import binom, norm

def groups_f_stat(*groups, axis=0):
    f_stat, pval = f_oneway(*groups, axis=axis)
    # replace NaNs in the F statistic with zeros
    f_stat[np.isnan(f_stat)] = 0
    return f_stat


def kruskal_per_dim(*groups):
    assert len(groups) >= 2, "must provide at least two groups for comparison"
    # check that all groups data is 2d and has the same second dimension
    num_features = groups[0].shape[1]
    for g in groups:
        assert len(g.shape) == 2, "input must be 2d array"
        assert num_features == g.shape[1], "all groups must have the same second dimension"

    stats = np.empty((num_features,))
    pvals = np.empty_like(stats)
    for i in range(num_features):
        dim_i_groups = [g[:, i] for g in groups]
        stats[i], pvals[i] = kruskal(dim_i_groups)
    return stats, pvals


class FeatureSelectionBase(BaseEstimator, TransformerMixin):

    def __init__(self, verbosity=False):
        self.selected_features = None
        self.verbosity = verbosity
        self.eps = 1e-20

    def fit(self, X, y):
        """
        Args:
        :param X:  data with shape (n_samples, n_features)
        :param y:  target values with shape (n_samples,)
        Returns:
        self: object
        """
        self.selected_features = np.arange(0, X.shape[1])
        return self

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
            return X[:, self.selected_features]

    # def fit_transform(self, X, y=None):
    #    if self.selected_features is None:
    #        self.fit(X,y)
    #    # transform the data
    #    return self.transform(X)

    def get_num_selected_features(self):
        if self.selected_features is None:
            assert False, "selected features property is still None, this method must be called after fit"
        else:
            return self.selected_features.shape[0]


class MultiTestCdfNorm(MultiTest):
    def __init__(self, pvals):
        super().__init__(pvals)
        assert len(pvals) > 1, "in MultiTestCdfNorm, length of passed p-values vector must be greater than 1"
        assert not np.isinf(pvals).any(), "in MultiTestCdfNorm, passed p-values vector has infinite values"
        assert not np.isnan(pvals).any(), "in MultiTestCdfNorm, passed p-values vector has null values"
        self._pvals = np.sort(np.unique(pvals.copy()))
        self._N = len(self._pvals)
        #self._uu = np.linspace(np.min(self._pvals), np.max(self._pvals), self._N)
        self._uu = np.linspace(1 / self._N, 1, self._N)

        #bins = np.histogram_bin_edges(pvals, bins=self._N, range=(0.0, 1.0), weights=None)
        hist, bins = np.histogram(self._pvals, bins=[0]+list(self._uu))
        self.cdf = np.cumsum((1/self._N) * hist)

        # version 1
        #self._zz = np.sqrt(self._N) * (self._uu - self._pvals) / cdf

        self.hc_obj_pvals = ((self._uu[:-1] - self._pvals[:-1])/np.sqrt(self._pvals[:-1] * (1-self._pvals[:-1])))
        self.hc_obj_cdf = (self.cdf[:-1] - self._uu[:-1]) / np.sqrt(self.cdf[:-1] * (1-self.cdf[:-1]))
        self.pvals_inv_std = 1/np.sqrt(self._pvals[:-1] * (1-self._pvals[:-1]))

        # version 2
        self._zz = np.sqrt(self._N) * self.pvals_inv_std * self.hc_obj_cdf
        #self._zz = np.sqrt(self._N) * np.sqrt((self.cdf[:-1] - self._uu[:-1])**2) / np.sqrt(self.cdf[:-1] * (1-self.cdf[:-1]))
        self._imin_star = np.argmax(self._pvals > (1 - 1/self._N) / self._N)
        self._imin_jin = np.argmax(self._pvals > np.log(self._N) / self._N)

    def _calculate_hc(self, imin, imax):
        if imin > imax:
            return np.nan
        if imin == imax:
            self.i_cdf = imin
        else:
            self.i_cdf = np.argmax(self._zz[imin:imax]) + imin
        zMaxStar = self._zz[self.i_cdf]
        self._istar = int(self._N * self.cdf[self.i_cdf])
        return zMaxStar, self._pvals[self._istar]

class hc2(FeatureSelectionBase):
    def __init__(self, feature_selector, num_resampling: int, num_samples_per_class: int, hc_gamma=0.2, hc_stbl=False, hc_method='star'):
        super().__init__()
        self.feature_selector = feature_selector
        self.num_resampling = num_resampling
        self.num_samples_per_class = num_samples_per_class
        self.hc_gamma = hc_gamma
        self.hc_stbl = hc_stbl
        self.hc_method = hc_method
        self.mt = None
        self.hc_obj = None
        self.hc = None
        self.hct = None
        self.selected_features = None
        self.selection_pvalues = None
        self.num_features_selected_in_sample = []


    def fit(self, X, y):
        self.num_selections = np.zeros((X.shape[1],))
        self.num_features_selected_in_sample = []
        # using bootstrap (resampling with replacement), generate multiple datasets from the given single dataset
        cls_idx = {label : np.flatnonzero(y == label) for label in list(np.unique(y))}

        for i in range(self.num_resampling):
            X_bootstrap = np.concatenate([X[np.random.choice(cls_idx[label], size=self.num_samples_per_class, replace=False), :] for label in cls_idx.keys()])
            y_bootstrap = np.concatenate([np.repeat(label, self.num_samples_per_class) for label in cls_idx.keys()])
            fs = self.feature_selector(hc_gamma=self.hc_gamma, hc_stbl=self.hc_stbl, hc_method=self.hc_method)
            fs.fit(X_bootstrap, y_bootstrap)
            if len(fs.selected_features) < X.shape[1]:
                self.num_selections[fs.selected_features] += 1
                self.num_features_selected_in_sample.append(len(fs.selected_features))

        # approximating a poisson-binomial distribution with normal variables
        experiments_success_prob = np.array(self.num_features_selected_in_sample)/X.shape[1]
        self.mu = np.sum(experiments_success_prob)
        self.std = np.sqrt(np.sum(experiments_success_prob * (1-experiments_success_prob)))
        #binomial_rv = binom(n=self.num_resampling, p=self.hc_gamma)
        normal_dist = norm(loc=self.mu, scale=self.std)
        self.selection_pvalues = normal_dist.sf(self.num_selections)
        self.mt = MultiTest(self.selection_pvalues, stbl=self.hc_stbl)
        self.hc_obj = self.mt._zz
        
        if self.hc_method == 'star':
            self.hc, self.hct = self.mt.hc_star(gamma=self.hc_gamma)
        elif self.hc_method == 'jin':
            self.hc, self.hct = self.mt.hc_jin(gamma=self.hc_gamma)
        elif self.hc_method == 'hc':
            self.hc, self.hct = self.mt.hc(gamma=self.hc_gamma)

        self.selected_features = np.flatnonzero(self.selection_pvalues <= self.hct)
        return self
        

class FeatureSelectionDiversityPursuitAnova(FeatureSelectionBase):

    def __init__(self, hc_gamma=0.2, hc_stbl=True, hc_method='star',
                 use_emp_cdf_in_hc_obj: bool = False,
                 override_inf_nan_stat=False,
                 num_null_pvals_vectors: int = 1000,
                 transformer=None, verbosity=False):
        super().__init__()
        self.test = f_oneway
        self.hc_gamma = hc_gamma
        self.hc_stbl = hc_stbl
        self.hc_method = hc_method
        self.override_inf_nan_stat = override_inf_nan_stat
        self.transformer = transformer
        self.f_stat = None
        self.pvals = None
        self.mask = None
        self.hc = None
        self.hct = 1
        self.hc_obj = None
        self.num_null_pvals_vectors = num_null_pvals_vectors
        self.use_emp_cdf_in_hc_obj = use_emp_cdf_in_hc_obj
        self.num_features = None
        self.verbosity = verbosity
        assert self.hc_method in ['star', 'jin', 'hc', 'bj','take_all_below_gamma'], "HC method must be in ['hc', 'star', 'jin', 'bj','take_all_below_gamma']"

    def get_max_z_score_under_null(self, p, num_null_pvals_vectors, hc_stbl, hc_gamma):
        max_z_scores = []
        a = np.random.rand(num_null_pvals_vectors, p)
        for i in range(num_null_pvals_vectors):
            mt = MultiTest(a[i, :], stbl=hc_stbl)
            max_z_score, _ = mt.hc_star(gamma=hc_gamma)
            max_z_scores.append(max_z_score)
        valid_z_score_thd = np.percentile(max_z_scores, 50)
        return max_z_scores, valid_z_score_thd


    def apply_hc(self, pvalues):
        """
        Applies higher criticism thresholding on the uni-variate features P-values
        Returns:
        self : object
        """
        if self.use_emp_cdf_in_hc_obj:

            assert False, "use_emp_cdf_in_hc_obj not supported"
        else:
            self.mt = MultiTest(pvalues, stbl=self.hc_stbl)

        # calculate a distribution of z_scores (maximums of the HC objective) we get under the null hypothesis that no feature is informative
        if self.hc_method != 'bj':
            max_z_scores, valid_z_score_thd = self.get_max_z_score_under_null(p=pvalues.shape[0],
                                                                              num_null_pvals_vectors=self.num_null_pvals_vectors,
                                                                              hc_stbl=self.hc_stbl,
                                                                              hc_gamma=self.hc_gamma)
        self.hc_obj = self.mt._zz
        gamma_idx = int(self.hc_gamma * self.num_features)
        if self.hc_method == 'star':
            self.hc, self.hct = self.mt.hc_star(gamma=self.hc_gamma)
        elif self.hc_method == 'jin':
            self.hc, self.hct = self.mt.hc_jin(gamma=self.hc_gamma)
        elif self.hc_method == 'hc':
            self.hc, self.hct = self.mt.hc(gamma=self.hc_gamma)
        elif self.hc_method == 'bj':
            self.hct = self.mt.berkjones_threshold(gamma=self.hc_gamma)
        elif self.hc_method == 'take_all_below_gamma':
            self.hc = valid_z_score_thd + 1
            self.hct = self.mt._pvals[gamma_idx]

        # check that the maximal z_score we got for the HC objective is meaningful - i.e. that it's not likely to get it under the null
        # if it's smaller than the value calculated above for then null distribution, override the calculated HCT and set it to 1 (take all the features)
        if self.hc is not None:
            if self.hc < valid_z_score_thd:
                self.hct = 1
        return self.hct

    def fit(self, X, y):
        samples = [X[y == label, :] for label in list(np.unique(y))]
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
        # update stat/pval vectors for all other features, where the f_stat is finite - these are the features
        # we're going to apply feature selection on.
        self.pvals[features_for_test_idx] = features_for_test_pvals
        self.f_stat[features_for_test_idx] = features_for_test_f_stat
        # apply higher criticism - notice we do it only on features with P-values different from 0,1
        hct = self.apply_hc(np.unique(features_for_test_pvals))
        mask = (features_for_test_pvals <= hct)
        # the returning mask has non-zero entries in indices aligned to the number of features we applied HC on (only features with finite stat!)
        # need to get the indices of these features in the original (full) feature list
        hc_selected_features_idx = features_for_test_idx[np.flatnonzero(mask)]
        # update the feature selector mask with the hc features
        self.mask[hc_selected_features_idx] = 1
        # if no features are selected, return all features
        if np.count_nonzero(self.mask) == 0:
            self.selected_features = np.arange(0, X.shape[1])
        else:
            self.selected_features = np.flatnonzero(self.mask)
        return self

    def get_features_for_test(self, groups):
        # first, compute the statistic for all features
        all_features_f_stat, _ = f_oneway(*groups, axis=0)
        # find where the f_stat is infinite - these are features we'll definitely want to take, since their "within"
        # variance is 0
        inf_f_stat_idx = np.isinf(all_features_f_stat)
        # find where the f_stat is undefined - these are degenerated features
        nan_f_stat_idx = np.isnan(all_features_f_stat)
        # get indices for the rest of the features
        features_for_test_idx = np.array(
            list(set(range(self.num_features)) - set(np.flatnonzero(inf_f_stat_idx)) - set(np.flatnonzero(nan_f_stat_idx))))
        if self.verbosity:
            print(f"In feature selector {self.__class__.__name__}, number of inf features = {np.count_nonzero(inf_f_stat_idx)}")
            print(f"In feature selector {self.__class__.__name__}, number of nan features = {np.count_nonzero(nan_f_stat_idx)}")

        # features_for_test_idx = np.array(list(set(range(self.num_features)) - set(np.flatnonzero(nan_f_stat_idx))))
        return inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx

    def test_func(self, groups):

        if self.override_inf_nan_stat:
            inf_f_stat_idx = []
            nan_f_stat_idx = []
            features_for_test_idx = np.array(list(range(self.num_features)))
        else:
            inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx = self.get_features_for_test(groups)

        if self.verbosity:
            print(f"In feature selector {self.__class__.__name__}, number of features for hc test = {len(features_for_test_idx)}")
        groups_for_test = [group[:, features_for_test_idx] for group in groups]
        if self.transformer is not None:
            groups_for_test = [self.transformer.fit_transform(group) for group in groups_for_test]

        features_for_test_f_stat, features_for_test_pvals = f_oneway(*groups_for_test)

        if self.override_inf_nan_stat:
            features_for_test_pvals[np.isinf(features_for_test_f_stat)] = self.eps
            features_for_test_pvals[np.isnan(features_for_test_f_stat)] = 1 - self.eps

        return features_for_test_f_stat, features_for_test_pvals, inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx


class FeatureSelectionDiversityPursuitPermutation(FeatureSelectionDiversityPursuitAnova):

    def __init__(self, n_resamples, hc_gamma=0.2,
                 hc_stbl=True, hc_method='star', use_emp_cdf_in_hc_obj: bool = False, override_inf_nan_stat=False,
                 transformer=None, verbosity=False):
        super().__init__(hc_gamma=hc_gamma, hc_stbl=hc_stbl, hc_method=hc_method)
        self.n_resamples = n_resamples
        self.null_dist = {}

    def test_func(self, groups):
        inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx = self.get_features_for_test(groups)
        groups_for_test = [group[:, features_for_test_idx] for group in groups]
        if self.transformer is not None:
            groups_for_test = [self.transformer.fit_transform(group) for group in groups_for_test]
        res = permutation_test(groups_for_test, groups_f_stat, n_resamples=self.n_resamples, vectorized=True, alternative='greater',
                               permutation_type='independent')
        features_for_test_f_stat, features_for_test_pvals = res.statistic, res.pvalue
        return features_for_test_f_stat, features_for_test_pvals, inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx


class FeatureSelectionDiversityPursuitKruskal(FeatureSelectionDiversityPursuitAnova):

    def get_features_for_test(self, groups):

        # when all data is identical, the kruskal function will raise a value error. meaning for us that the feature is uninformative.
        # that is equivalent to the ANOVA giving nan value. find features where all data is identical
        X = np.concatenate(groups, axis=0)
        nan_f_stat_idx = (np.std(X, axis=0) == 0)

        # unlike the ANOVA F stat, the kruskal H statistic cannot be infinite when at least some of the observations are different.
        inf_f_stat_idx = np.zeros((X.shape[1],), dtype=bool)

        # get indices for the rest of the features
        features_for_test_idx = np.array(
            list(set(range(self.num_features)) - set(np.flatnonzero(inf_f_stat_idx)) - set(np.flatnonzero(nan_f_stat_idx))))
        if self.verbosity:
            print(f"In feature selector {self.__class__.__name__}, number of inf features = {np.count_nonzero(inf_f_stat_idx)}")
            print(f"In feature selector {self.__class__.__name__}, number of nan features = {np.count_nonzero(nan_f_stat_idx)}")

        # features_for_test_idx = np.array(list(set(range(self.num_features)) - set(np.flatnonzero(nan_f_stat_idx))))
        return inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx

    def test_func(self, groups):
        if self.override_inf_nan_stat:
            inf_f_stat_idx = []
            nan_f_stat_idx = []
            features_for_test_idx = np.array(list(range(self.num_features)))
        else:
            inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx = self.get_features_for_test(groups)
        if self.verbosity:
            print(f"In feature selector {self.__class__.__name__}, number of features for hc test = {len(features_for_test_idx)}")
        num_features = groups[0].shape[1]
        # init arrays for f_stat, pvals
        f_stat = np.zeros((num_features,))
        pvals = np.ones((num_features,))
        for i in range(len(features_for_test_idx)):
            current_feature = features_for_test_idx[i]
            samples = [np.squeeze(group[:, current_feature]) for group in groups]
            if self.transformer is not None:
                samples = [np.squeeze(self.transformer.fit_transform(sample.reshape(-1, 1))) for sample in samples]
            try:
                f_stat[current_feature], pvals[current_feature] = kruskal(*samples)
            except ValueError:
                # when all data is identical, the kruskal function will raise a value error. meaning for us that the feature is uninformative.
                # that is equivalent to the ANOVA giving nan value
                f_stat[current_feature] = 0
                pvals[current_feature] = 1 - self.eps

        features_for_test_f_stat = f_stat[features_for_test_idx]
        features_for_test_pvals = pvals[features_for_test_idx]

        return features_for_test_f_stat, features_for_test_pvals, inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx


class FeatureSelectionDiversityPursuitAnderKsamp(FeatureSelectionDiversityPursuitAnova):

    def test_func(self, groups):
        num_features = groups[0].shape[1]
        # init arrays for f_stat, pvals
        f_stat = np.zeros((num_features,))
        pvals = np.ones((num_features,))
        for j in range(num_features):
            samples = [np.squeeze(group[:, j]) for group in groups]
            res = anderson_ksamp(samples)

            f_stat[j], pvals[j] = res.statistic, res.pvalue

        # The anderson statistic is capped at 0.25, so there cannot be infinite values in the stat array
        inf_f_stat_idx = np.zeros((num_features,), dtype=bool)
        # get the indice of the "nan" H statistic
        nan_f_stat_idx = np.flatnonzero((pvals == 1))
        features_for_test_idx = np.array(
            list(set(range(self.num_features)) - set(np.flatnonzero(inf_f_stat_idx)) - set(np.flatnonzero(nan_f_stat_idx))))
        features_for_test_f_stat = f_stat[features_for_test_idx]
        features_for_test_pvals = pvals[features_for_test_idx]
        return features_for_test_f_stat, features_for_test_pvals, inf_f_stat_idx, nan_f_stat_idx, features_for_test_idx


class FeatureSelectionOneVsAllAnova(FeatureSelectionDiversityPursuitAnova):

    def fit(self, X, y):
        # init the feature mask to all zeros (i.e. no features are selected)
        self.mask = np.zeros((X.shape[1],))
        self.num_features = X.shape[1]
        # collect all the stat/p-values in a dict
        self.f_stat = {t: np.ones_like(self.mask) for t in list(np.unique(y))}
        self.pvals = {t: np.ones_like(self.mask) for t in list(np.unique(y))}
        # go over all the targets
        for t in list(np.unique(y)):
            # get all samples with the current target as one group
            g1 = X[y == t, :]
            # get the rest of the samples (corresponding to all other targets) as the second group
            g2 = X[y != t, :]
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
            hct = self.apply_hc(features_for_test_pvals)
            mask = (features_for_test_pvals <= hct)
            hc_selected_features_idx = features_for_test_idx[np.flatnonzero(mask)]
            # update the feature selector mask with the hc features
            self.mask[hc_selected_features_idx] = 1

        # finally, get the features indices from the mask
        # if no features are selected, return all features
        if np.count_nonzero(self.mask) == 0:
            self.selected_features = np.arange(0, X.shape[1])
        else:
            self.selected_features = np.flatnonzero(self.mask)
        return self


class FeatureSelectionOneVsAllKS(FeatureSelectionOneVsAllAnova):
    def test_func(self, groups):
        num_features = groups[0].shape[1]
        # init arrays for f_stat, pvals
        stat = np.zeros((num_features,))
        pvals = np.ones((num_features,))
        for j in range(num_features):
            samples = [np.squeeze(group[:, j]) for group in groups]
            res = ks_2samp(*samples)
            stat[j], pvals[j] = res.statistic, res.pvalue

        inf_stat_idx = np.isinf(stat)
        nan_stat_idx = np.isnan(stat)
        features_for_test_idx = np.array(list(set(range(self.num_features)) - set(inf_stat_idx) - set(nan_stat_idx)))
        features_for_test_stat = stat[features_for_test_idx]
        features_for_test_pvals = pvals[features_for_test_idx]
        return features_for_test_stat, features_for_test_pvals, inf_stat_idx, nan_stat_idx, features_for_test_idx
