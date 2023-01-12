import os
import sys
import scipy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from scipy.stats import f as fdist
from scipy.stats import t as tdist
from scipy.stats import norm
from scipy.stats import ttest_ind as ttest

from twosample import bin_allocation_test
from multitest import MultiTest
from typing import List

from sklearn.metrics import pairwise_distances

class CentroidSimilarity(object):
    """
    Classify based on most similar centroid.

    At training, we average the response of each feature over
    classes. We store the class centroids (averages).

    At prediction, we give the highest probability to the class
    that is most similar to the test sample (in Eucleadian distance
    or cosine similarity).

    """

    def __init__(self,hc_gamma=0.2,print_shapes=False):
        self.global_mean = None
        self.classes = None
        self.cls_mean = None
        self.mask = None
        self.gamma = hc_gamma
        self.num_selected_features = 0
        self.print_shapes = print_shapes

    def fit(self, X, y):
        """

        Args:
        :param X:  training data as a matrix n X p of n samples with p features each
        :param y:  labels. For multi-label, simply pass the same x value with different labels.

        """
        self.classes = np.unique(y)
        self.global_mean = np.mean(X, 0)
        self.global_std = np.std(X, 0)

        self.cls_mean = np.zeros((len(self.classes), X.shape[1]))
        self.cls_std = np.zeros((len(self.classes), X.shape[1]))
        self.cls_n = np.zeros((len(self.classes), X.shape[1]))

        for i, _ in enumerate(self.classes):
            x_cls = X[y == self.classes[i]]
            self.cls_mean[i] = np.mean(x_cls, 0)
            self.cls_std[i] = np.std(x_cls, 0)
            self.cls_n[i] = len(x_cls)
        # in this basic class, all features are taken into account, hence the mask is just a vector of 1's
        self.set_mask(np.ones_like(self.cls_mean))

    def set_mask(self, mask):
        """
        :param mask: a binary array of shape (k,p) where k,p are the number of classes,features respectively
        indicating which features should be considered in classifying each class
        """
        # apply the mask - perform an element-wise multiplication between the raw class centroids and the mask
        means = self.cls_mean * mask
        self.mask = mask
        # compute the classifier centroids
        self.mat = (means.T / np.linalg.norm(means, axis=1)).T
        

    def prob_func(self, response):
        return np.exp(response) / (1 + np.exp(response))

    def get_centroids(self):
        return self.mat * self.mask

    def predict_log_proba(self, X):
        # perform an inner product of the input with the centroids. since the centroids are normalized to have unit length,
        # this is equivalent to a cosine similarity between the input and the centroids
        response = X @ self.get_centroids().T
        return self.prob_func(response)

    def predict(self, X):
        probs = self.predict_log_proba(X)
        return self.classes[np.argmax(probs, 1)]  # max inner product

    def eval_accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_mask_prec_recall(self, true_mask):
        """
        find the false discovery rate of the features.
        FDR = 1 - precision
        """
        TP = np.sum(np.logical_and(self.mask,true_mask))
        FP = np.sum(np.logical_and(self.mask,np.logical_not(true_mask)))
        FN = np.sum(np.logical_and(np.logical_not(self.mask),true_mask))
        precision = TP / (FP + TP)
        recall = TP / (TP + FN)
        
        return recall, precision



class CentroidSimilarityFeatureSelection(CentroidSimilarity):
    """
    Same as CentroidSimilarity, but now we mask
    some of the features based on two methods:
        'one_vs_all':  two-sample test
        'diversity_pursuit' ('all vs all'): full one-way ANOVA

    """

    def fit(self, X, y, method='one_vs_all'):

        super().fit(X, y)
        self.cls_response = np.zeros(len(self.classes))
        mask = np.ones_like(self.cls_mean)

        for i, cls in enumerate(self.classes):
            mask[i] = self.get_cls_mask(i, method=method)

        self.set_mask(mask)

    def get_pvals(self, cls_id, method='one_vs_all'):
        """
        compute P-values associated with each feature
        for the given class
        """

        mu1 = self.cls_mean[cls_id]
        n1 = self.cls_n[cls_id]
        std1 = self.cls_std[cls_id]
        # get the total number of samples by summing the number of samples in each class
        nG = self.cls_n.sum(0)
        stdG = self.global_std
        muG = self.global_mean

        assert (method in ['one_vs_all', 'diversity_pursuit'])
        if method == 'one_vs_all':
            pvals, _, _ = one_vs_all_anova(n1, nG, mu1, muG, std1, stdG)
        if method == 'diversity_pursuit':
            pvals, _, _ = diversity_pursuit_anova(self.cls_n,
                                                  self.cls_mean,
                                                  self.cls_std,
                                                  self.print_shapes)
        return pvals

    def get_cls_mask(self, cls_id, method='one_vs_all'):
        """
        this function computes a class feature mask. it first produces a p-value for each feature
        to determine the feature ability to separate the class from other classes. it then applies higher criticism 
        to the array of p-values to select the most important features
        :param cls_id: id of the class we'd like to test
        :param method: method to compute the p-values. select from {one_vs_all,diversity_pursuit}
        returns:
        mask - a binary vector with shape (1,p) where p is the number of features 
        """
        pvals = self.get_pvals(cls_id, method=method)

        mt = MultiTest(pvals)
        hc, hct = mt.hc_star(gamma=self.gamma)
        self.cls_response[cls_id] = hc
        mask = pvals < hct
        return mask

def diversity_pursuit_anova(classes_num_samples, cls_mean, cls_std, print_shapes=False):
    """
    F-test for discovering discriminating features
    The test is vectorized along the last dimension where different entries corresponds to different features

    Args:
    -----
    :param classes_num_samples:  vector indicating the number of elements in each class
    :param cls_mean:  matrix of classes means with shape (k,p); the (i,j) entry is the class i mean in feature j
    :param cls_std:  matrix of standard errors; the (i,j) entry is the standard
          error of class i in feature j

    """
    # compute the per feature global mean - the mean of all the data for each feature. shape = (1,p)
    global_mean = np.expand_dims(np.sum(cls_mean * classes_num_samples, 0) / np.sum(classes_num_samples, 0),axis=0)
    # compute the "between groups" sum of squares
    SSbetween = np.sum(classes_num_samples * (cls_mean - global_mean) ** 2, 0)
    SSwithin = np.sum((classes_num_samples-1) * (cls_std ** 2), 0)
    if print_shapes:
        print("in function diversity_pursuit_anova:")
        print(f"classes_num_samples.shape = {classes_num_samples.shape}")
        print(f"cls_std.shape = {cls_std.shape}")
        print(f"cls_mean.shape = {cls_mean.shape}")
        print(f"global_mean.shape = {global_mean.shape}")
        print(f"SSwithin.shape = {SSwithin.shape}")
        print(f"SSbetween.shape = {SSbetween.shape}")
    
    # the numerator number of degrees of freedom is actually k-1 (k is the number of groups/classes)
    dfn = len(classes_num_samples) - 1
    # the denominator number of degrees of freedom is n-k (n is the total number of samples)
    dfd = np.sum(classes_num_samples, 0) - len(classes_num_samples)

    f_stat = (SSbetween / dfn) / (SSwithin / dfd)
    return fdist.sf(f_stat, dfn, dfd), SSbetween, SSwithin


def one_vs_all_anova(n1, nG, mu1, muG, std1, stdG):
    """
    :param n1: number of samples from the isolated class
    :param nG: total number of samples (all classes)
    :param mu1: mean response of the isolated class samples
    :param muG: mean response of all samples
    :param std1: standard deviation of the isolated class samples response
    :param stdG: standard deviation of all samples response
    :return:
    """
    # get the number of samples in all other classes except the isolated class
    n2 = nG - n1
    # get the mean response in all other classes except the isolated class
    mu2 = (muG * nG - mu1 * n1) / (nG - n1)
    # compute the "between groups" sum of squares
    SSbetween = n1 * (mu1 - muG) ** 2 + n2 * (mu2 - muG) ** 2
    SStot = stdG ** 2 * (nG - 1)
    # using the sum of squares decomposition to compute the variability within each group
    SSwithin = SStot - SSbetween
    # calculate the F-statistic
    f_stat = (SSbetween / 1) / (SSwithin / (nG - 2))
    return fdist.sf(f_stat, 1, nG - 2), SSbetween, SSwithin
