import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax
from scipy.stats import f as fdist
from multitest import MultiTest
from sklearn.preprocessing import normalize

class CentroidSimilarity(object):
    """
    Classify based on most similar centroid.

    At training, we average the response of each feature over
    classes. We store the class centroids (averages).

    At prediction, we give the highest probability to the class
    that is most similar to the test sample (in Euclidian distance
    or cosine similarity).

    """

    def __init__(self,hc_gamma=0.2, use_euclidian_distance=False, print_shapes=False):
        self.global_mean = None
        self.classes = None
        self.cls_mean = None
        self.mask = None
        self.gamma = hc_gamma
        self.num_selected_features = 0
        self.print_shapes = print_shapes
        self.use_euclidian_distance = use_euclidian_distance

    def fit(self, X, y):
        """

        Args:
        :param X:  training data with shape (n_samples, n_features)
        :param y:  target values with shape (n_samples,)

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
        self.centroids = normalize(means, norm='l2')
        

    def sigmoid(self, response):
        return np.exp(response) / (1 + np.exp(response))

    def get_centroids(self):
        return self.centroids # * self.mask

    def get_dist_from_centroids(self, X):
        """
        computes the euclidian distance of the instances (point) of input X from the classifier centroids
        Args:
            X: ndarray with shape (n_samples, n_features)

        Returns:
            distances - ndarray of shape (n_samples_X, num_classes)

        """
        centroids = self.cls_mean * self.mask
        return euclidean_distances(X, centroids)

    def predict_log_proba(self, X):
        # perform an inner product of the input with the centroids. since the centroids are normalized to have unit length,
        # this is equivalent to a cosine similarity between the input and the centroids
        response = X @ self.get_centroids().T
        return self.sigmoid(response)

    def predict(self, X):
        if self.use_euclidian_distance:
            distances = self.get_dist_from_centroids(X)
            probs = softmax(distances, axis=1)
        else:
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

        assert (method in ['one_vs_all', 'diversity_pursuit'])
        if method == 'one_vs_all':
            pvals, _, _ = self.one_vs_all_anova(num_smpls_in_cls=self.cls_n[cls_id],
                                                total_num_smpls=self.cls_n.sum(0),
                                                cls_mean=self.cls_mean[cls_id],
                                                global_mean=self.global_mean,
                                                global_std=self.global_std)
        if method == 'diversity_pursuit':
            pvals, _, _ = self.diversity_pursuit_anova(self.cls_n,
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
        if method == 'union':
            one_vs_all_mask = self.get_cls_mask(cls_id=cls_id, method='one_vs_all')
            diversity_pursuit_mask = self.get_cls_mask(cls_id=cls_id, method='diversity_pursuit')
            mask = np.logical_or(one_vs_all_mask, diversity_pursuit_mask)
        elif method == 'diversity_pursuit_no_ova':
            one_vs_all_mask = self.get_cls_mask(cls_id=cls_id, method='one_vs_all')
            diversity_pursuit_mask = self.get_cls_mask(cls_id=cls_id, method='diversity_pursuit')
            mask = np.logical_and(np.logical_not(one_vs_all_mask), diversity_pursuit_mask)

        else:
            pvals = self.get_pvals(cls_id, method=method)
            mt = MultiTest(pvals)
            self.hc, self.hct = mt.hc_star(gamma=self.gamma)
            self.cls_response[cls_id] = self.hc
            mask = pvals < self.hct
        return mask

    def diversity_pursuit_anova(self,classes_num_samples, cls_mean, cls_std, print_shapes=False):
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
        #global_mean = np.expand_dims(np.sum(cls_mean * classes_num_samples, 0) / np.sum(classes_num_samples, 0),axis=0)
        # compute the "between groups" sum of squares
        SSbetween = np.sum(classes_num_samples * (cls_mean - self.global_mean) ** 2, 0)
        SSwithin = np.sum((classes_num_samples-1) * (cls_std ** 2), 0)
        if print_shapes:
            print("in function diversity_pursuit_anova:")
            print(f"classes_num_samples.shape = {classes_num_samples.shape}")
            print(f"cls_std.shape = {cls_std.shape}")
            print(f"cls_mean.shape = {cls_mean.shape}")
            print(f"global_mean.shape = {self.global_mean.shape}")
            print(f"SSwithin.shape = {SSwithin.shape}")
            print(f"SSbetween.shape = {SSbetween.shape}")

        # the numerator number of degrees of freedom is actually k-1 (k is the number of groups/classes)
        dfn = len(classes_num_samples) - 1
        # the denominator number of degrees of freedom is n-k (n is the total number of samples)
        dfd = np.sum(classes_num_samples, 0) - len(classes_num_samples)
        # initialize the F statistics vector to 0 (for all features)
        f_stat = np.zeros_like(SSbetween)
        # run over all features and identify points where infinity will occur - assign an infinite F statistic there
        for i in range(SSwithin.shape[0]):
            if SSwithin[i] == 0 and SSbetween[i] > 0:
                f_stat[i] = 1e8

        f_stat = np.divide((SSbetween / dfn), (SSwithin / dfd), out=f_stat, where=(SSwithin != 0))
        return fdist.sf(f_stat, dfn, dfd), SSbetween, SSwithin


    def one_vs_all_anova(self,num_smpls_in_cls, total_num_smpls, cls_mean, global_mean, global_std):
        """
        In this method the data is partitioned to two groups: one with samples from a given class (the isolated class), and one part with the rest
        of the samples. Then, a one-way ANOVA test is performed for each feature between these two groups.
        :param num_smpls_in_cls: number of samples from the isolated class
        :param total_num_smpls: total number of samples (all classes)
        :param cls_mean: mean of the isolated class samples (basically the un-normalized class centroid)
        :param global_mean: global mean of all samples in the data
        :param global_std: standard deviation of each feature, over all samples in the data
        :return:
        """
        # get the number of samples in all other classes except the isolated class
        num_smpls_in_rest = total_num_smpls - num_smpls_in_cls
        # get the mean response in all other classes except the isolated class
        mean_smpls_in_rest = (global_mean * total_num_smpls - cls_mean * num_smpls_in_cls) / num_smpls_in_rest
        # compute the "between groups" sum of squares
        SSbetween = num_smpls_in_cls * (cls_mean - global_mean) ** 2 + num_smpls_in_rest * (mean_smpls_in_rest - global_mean) ** 2
        SStot = global_std ** 2 * (total_num_smpls - 1)
        # using the sum of squares decomposition to compute the variability within each group
        SSwithin = SStot - SSbetween
        # calculate the F-statistic
        assert (total_num_smpls > 2).all(), "total number of samples must be greater then number of groups"
        # initialize the F statistics vector to 0 (for all features)
        f_stat = np.zeros_like(SSbetween)
        # run over all features and identify points where infinity will occur - assign an infinite F statistic there
        for i in range(SSwithin.shape[0]):
            if SSwithin[i] == 0 and SSbetween[i] > 0:
                f_stat[i] = 1e8

        f_stat = np.divide(SSbetween,(SSwithin / (total_num_smpls - 2)), out=f_stat, where=(SSwithin!=0))
        return fdist.sf(f_stat, 1, total_num_smpls - 2), SSbetween, SSwithin
