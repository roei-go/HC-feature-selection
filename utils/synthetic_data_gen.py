import numpy as np

def sample_centroids(num_classes, num_features, eps, power, non_nulls_location='free'):
    """
    Randomly sample sparse class means
    
    Args:
    -----
    :param num_classes:   number of classes
    :param num_features:   number of features
    :param eps: fraction of non-null features
    :param power: amplitude of non-null features
    :param non_nulls_location:   if 'free', each class has different 
                            set of non-null features
                            if 'fixed', non-null features are
                            the same across all classes
    """
    if non_nulls_location == 'fixed':
        idcs = np.random.rand(num_features) < eps
        
    centroids = np.zeros((num_classes, num_features))
    # running over classes
    for i in range(num_classes):
        # randomly set the indices of non-null features
        if non_nulls_location == 'free':
            idcs = np.random.rand(num_features) < eps
        # make some of the non-null features negative and some positive
        centroids[i][idcs] = power * (1-2*(np.random.rand(np.sum(idcs)) > .5)) / 2
    return centroids


def sample_normal_clusters(centroids, n, sigma):
    """
    Sample noisy data from class means matrix
    
    Args:
    :centroids: class means matrix
    :n:   number of samples
    :sigma: noise standard deviation
    
    """
    c, p = centroids.shape
    Z = np.random.randn(n, p)
    y = np.random.randint(c, size=n)
    X = centroids[y] + sigma * Z
    return X, y
