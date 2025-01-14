from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from utils.hyper_parameters_tuning import model_hypopt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from collections import Counter
import plotly
import sklearn
from lib.classifier_with_feature_selection import ClassifierFeatureSelection
from scipy.spatial.distance import cdist


def classifiers_hyper_tune(classifiers: list,
                           search_spaces: list,
                           X,
                           y,
                           num_classes: int = 5,
                           num_train_samples_per_class: int = 10,
                           iterations: int = 100):
    classes_in_data = list(np.unique(y))
    if num_classes == len(classes_in_data):
        label_set = classes_in_data
    elif num_classes < len(classes_in_data):
        # sample a label set
        label_set = np.random.choice(a=np.unique(y), size=num_classes, replace=False)
    else:
        raise ValueError("number of classes must be less or equal to actual number of classes in the data")
    # get train samples from each class
    smpl_idx = []
    for l in label_set:
        cls_idcs = np.squeeze(np.argwhere(y == l))
        smpl_idx.extend(list(np.random.choice(cls_idcs, num_train_samples_per_class, replace=False)))
    # loop over the classifiers and find hyper-parameters for each one
    best_params = []
    for i in range(len(classifiers)):
        hyper_opt = model_hypopt(model=classifiers[i],
                                 param_space=search_spaces[i],
                                 X=X[smpl_idx],
                                 y=y[smpl_idx],
                                 iterations=iterations)
        best_params.append(hyper_opt.run())
    return best_params


def scan_experiment(classifiers: list,
                    classifiers_names: list,
                    scan_arrays: dict,
                    X: np.ndarray,
                    y: np.ndarray,
                    num_experiments: int = 50,
                    score_function=accuracy_score,
                    preprocess_func: list = None):
    train_sizes = scan_arrays['train_sizes']
    num_classes = scan_arrays['num_classes']
    accuracies = np.empty((len(classifiers), len(train_sizes), len(num_classes), num_experiments))
    num_features = np.empty_like(accuracies)
    for i in range(len(train_sizes)):
        for j in range(len(num_classes)):
            print(f"Measuring accuracy with {num_classes[j]} classes,  {train_sizes[i]} training examples per class")
            print("---------------------------------------------------------------------------------------------------")
            for k in range(num_experiments):
                # sample a label set
                label_set = np.random.choice(a=np.unique(y), size=num_classes[j], replace=False)
                # sample a desired number of examples (feature vectors) for each class
                smpl_idx = []
                for l in label_set:
                    cls_idcs = np.squeeze(np.argwhere(y == l))
                    smpl_idx.extend(list(cls_idcs))
                # grab all the data with labels from the label set of this experiment and split to train and test
                X_train, X_test, y_train, y_test = train_test_split(X[smpl_idx],
                                                                    y[smpl_idx],
                                                                    train_size=train_sizes[i] * num_classes[j],
                                                                    stratify=y[smpl_idx])

                accuracies[:, i, j, k], num_features[:, i, j, k] = multiple_classifiers_fit_predict(classifiers=classifiers,
                                                                                                    X_train=X_train,
                                                                                                    y_train=y_train,
                                                                                                    X_test=X_test,
                                                                                                    y_test=y_test,
                                                                                                    score_func=score_function,
                                                                                                    preprocess_func=preprocess_func)

            for l in range(len(classifiers)):
                print(
                    f"classifier {classifiers_names[l]}, mean accuracy is {np.round(np.mean(accuracies[l, i, j, :]), 2)}, std error is {np.round(np.std(accuracies[l, i, j, :]) / np.sqrt(accuracies.shape[-1]), 2)}")
            print("\n\n")
            for l in range(len(classifiers)):
                print(
                    f"classifier {classifiers_names[l]}, mean num features is {np.round(np.mean(num_features[l, i, j, :]), 0)}, std error is {np.round(np.std(num_features[l, i, j, :]) / np.sqrt(num_features.shape[-1]), 0)}")
            print("\n\n")

    return np.squeeze(accuracies), np.squeeze(num_features)


def multiple_classifiers_fit_predict(classifiers,
                                     X_train, y_train,
                                     X_test, y_test,
                                     score_func=accuracy_score,
                                     preprocess_func: list = None,
                                     get_features: bool = False,
                                     scaler_func: object = None):
    """
    Args:
        classifiers: list of classifiers. Each classifier should provide a sklearn-like fit and predict functions
        X_train: train data with shape (n_samples, n_features)
        y_train: target values with shape (n_samples,)
        X_test: test data with shape (n_samples, n_features)
        y_test: target values with shape (n_samples,)
        score_func:

    Returns:
        scores: numpy array with shape (len(classifiers),) with the classifiers test scores
    """
    # init an empty array to hold the scores
    scores = np.empty((len(classifiers),))
    # init an array to hold the number of features used by each classifier - initialize to the data dimension
    num_features = np.ones((len(classifiers),)) * X_train.shape[1]
    selected_features = {}
    if preprocess_func is None:
        preprocess_func = [None for c in classifiers]
    for i in range(len(classifiers)):
        # in case the classifier has a feature selector, set the selected features attribute to None to make sure
        # the feature selector is re-fitted to the data
        if isinstance(classifiers[i], ClassifierFeatureSelection):
            if hasattr(classifiers[i].feature_selector, 'selected_features'):
                classifiers[i].feature_selector.selected_features = None
            if hasattr(classifiers[i].feature_selector, 'predict_random'):
                classifiers[i].predict_random = False

        if scaler_func is not None:
            scaler_func.fit(X_train)
            X_train_s = scaler_func.transform(X_train)
        else:
            X_train_s = X_train.copy()

        classifiers[i].fit(X_train_s, y_train)

        # get number of features from centroid similarity classifiers (which have a feature selector attribute)
        if isinstance(classifiers[i], ClassifierFeatureSelection):
            num_features[i] = classifiers[i].num_selected_features
            if get_features:
                if hasattr(classifiers[i].feature_selector, 'selected_features'):
                    selected_features[i] = list(classifiers[i].feature_selector.selected_features)

        # find number of features of sklearn pipelines often included in the experiment
        if isinstance(classifiers[i], sklearn.pipeline.Pipeline):
            # each pipeline has a 'steps' key, listing the pipe steps
            steps = [step[0] for step in classifiers[i].__dict__['steps']]
            # by convention, the classifier step in the pipeline is always keyed 'clf'
            assert 'clf' in steps, "pipeline must include classifier keyed clf"
            cls_name = classifiers[i]['clf'].__class__.__name__

            # if the pipeline includes a HC feature selector, get the number of features from it
            if 'feat_sel' in steps:
                num_features[i] = classifiers[i]['feat_sel'].get_num_selected_features()

            # for linear classifiers (logistic regression, linear SVC...)
            elif hasattr(classifiers[i]['clf'], 'coef_'):
                # find the number of features from the number of non-zero coefficients
                num_features[i] = np.count_nonzero(np.sum(np.abs(classifiers[i]['clf'].coef_), axis=0))
                selected_features[i] = list(np.flatnonzero(np.sum(np.abs(classifiers[i]['clf'].coef_), axis=0)))
            # most other sklearn classifiers has 'n_features_in_'
            elif hasattr(classifiers[i]['clf'], 'n_features_in_'):
                num_features[i] = classifiers[i]['clf'].n_features_in_

        if scaler_func is not None:
            X_test_s = scaler_func.transform(X_test)
        else:
            X_test_s = X_test.copy()
        y_pred = classifiers[i].predict(X_test_s)
        scores[i] = score_func(y_test, y_pred)

    if get_features:
        return scores, num_features, selected_features
    else:
        return scores, num_features


def power_transform(X, beta=0.5):
    eps = 1e-15
    Xb = np.power(X + eps, beta)
    Xb_norm = np.expand_dims(np.linalg.norm(Xb, axis=1), axis=1)
    # return normalize(Xb, axis=1)
    return Xb / Xb_norm


def preprocessing_model(input_shape, preprocess_layer, noise_model=None):
    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_layer(inputs)
    if noise_model is not None:
        outputs = noise_model(x)
    else:
        outputs = x
    return tf.keras.Model(inputs, outputs)


def identity_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    outputs = inputs
    return tf.keras.Model(inputs, outputs)


def get_feature_extractor(base_model, input_shape, pooling_layer):
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    outputs = pooling_layer(x)
    return tf.keras.Model(inputs, outputs)


def extract_features(ds, preprocessing_model, feature_extractor):
    X = []
    for batch_idx, batch in enumerate(ds):
        preprocessed_inputs = preprocessing_model(batch)
        features = feature_extractor(preprocessed_inputs).numpy()
        X.append(features)
    X = np.concatenate(X, axis=0)
    return X


def get_images_from_supervised_set(ds, label_set, img_size=[224, 224], max_samples_per_class=100):
    images = []
    labels = []
    samples_per_class = {label: 0 for label in label_set}
    # iterate over the data set, collect only samples belonging to the label set
    for image, label in tfds.as_numpy(ds):
        if label in label_set and samples_per_class[label] < max_samples_per_class:
            images.append(tf.image.resize(image, size=img_size, antialias=True))
            labels.append(label)
            samples_per_class[label] += 1

    # stack the images and labels into nd arrays
    images = tf.stack(images)
    labels = tf.stack(labels)
    return images, labels


def smallest_n_indices(a, n):
    idx = a.ravel().argsort()[:n]
    return np.stack(np.unravel_index(idx, a.shape)).T


def largest_n_indices(a, n):
    idx = a.ravel().argsort()[-n:]
    return np.stack(np.unravel_index(idx, a.shape)).T


def find_similar_embeddings(X, y, num_pairs: int = 10000, similarity='cosine'):
    if similarity == 'cosine':
        sim_mat = cosine_similarity(X)
    else:
        sim_mat = 1 / euclidean_distances(X)
    # get the lower triangle of the similarity matrix (without the main diagonal)
    sim_low_tri = np.tril(sim_mat, k=-1)
    # we want to find maximal elements - i.e. the most similar pairs.
    # replace all the zeros in the lower triangular matrix (the upper triangle is filled with 0) with large values
    sim_low_tri[sim_low_tri == 0] = np.min(sim_mat)
    # find the indices of the largest elements
    similar_pairs = largest_n_indices(sim_low_tri, num_pairs)
    # get the classes of the similar pairs in all pairs where the classes are different (i.e. when pair members don't belong in the same class)
    class_pairs = [sorted([y[similar_pairs[i, 0]], y[similar_pairs[i, 1]]]) for i in range(similar_pairs.shape[0]) if
                   y[similar_pairs[i, 0]] != y[similar_pairs[i, 1]]]
    # create a dictionary where the key is the pair of classes and the value is the number of times this pair appear in the list
    frequency_list = Counter(tuple(i) for i in class_pairs)
    # sort the dictionary by the frequency
    sorted_pair_by_frequency = sorted(frequency_list.items(), key=lambda x: x[1], reverse=True)
    return sorted_pair_by_frequency


def sample_equal_number_samples_from_class(X, y, num_train_samples_per_class):
    train_idcs, test_idcs = [], []
    label_set = list(np.unique(y))
    for l in label_set:
        # get indices of samples from the current class
        cls_idcs = np.squeeze(np.argwhere(y == l))
        # sample train samples from it
        cls_train_idcs = np.random.choice(cls_idcs, size=num_train_samples_per_class, replace=False)
        # the test samples are all samples not on the train
        cls_test_idcs = list(set(cls_idcs) - set(cls_train_idcs))
        train_idcs.extend(list(cls_train_idcs))
        test_idcs.extend(list(cls_test_idcs))
    X_train = X[train_idcs, :]
    y_train = y[train_idcs]
    X_test = X[test_idcs, :]
    y_test = y[test_idcs]
    return X_train, X_test, y_train, y_test


# create an additive noise
def add_noise(images, threshold, noise_delta, low_sat, high_sat):
    mask = np.random.rand(*images.numpy().shape[0:3])
    mask = mask < threshold
    mask = np.repeat(np.expand_dims(mask, axis=3), 3, axis=3)
    noise = np.random.randint(low=-noise_delta, high=noise_delta, size=images.shape) * mask
    return np.clip(images + noise, a_min=low_sat, a_max=high_sat)


def add_white_noise(images, gamma):
    # normalize images by max
    max_pix_val = np.amax(images)
    images = images / max_pix_val
    # generate white noise
    n = np.random.randn(*images.shape)
    images = gamma * images + (1 - gamma) * n
    images = max_pix_val * images
    return images


def resize_down_up(images, size):
    orig_size = [images.shape[1], images.shape[2]]
    small_images = tf.image.resize(images, size=size, antialias=True)
    return tf.image.resize(small_images, size=orig_size, antialias=True)


def open_html_report(html_path: str, file_title: str):
    f = open(html_path, 'a')
    f.write(f"<html><head><title>{file_title}</title></head>")
    f.write(f"<body>")
    return f


def close_html_report(f):
    f.write(f"</body>")
    f.write(f"</html>")
    f.close()


def gen_html_report(html_path: str, items: list):
    with open(html_path, 'a') as f:
        f.write("<html><head><title>HTML File</title></head>")
        f.write(f"<body>")
        for item in items:
            if isinstance(item, str):
                f.write(f"<p>{item}</p>")
            elif isinstance(item, plotly.graph_objs._figure.Figure):
                f.write(item.to_html(full_html=False, include_plotlyjs='cdn'))
            else:
                assert False, "Invalid item type"
        f.write(f"</body>")
        f.write(f"</html>")
        f.close()


def get_best_score_full_params(cv_results, param_values):
    # init a set with all grid points indices
    matching_points = set(range(len(cv_results['params'])))
    for k, v in param_values:
        matching_points = matching_points.intersection(set([i for i in range(len(cv_results['params'])) if cv_results['params'][i][k] == v]))

    matching_points = list(matching_points)
    # now we have a set of grid points that share a common set of param values. find which one has the best score and return it and also the parameters
    matching_points_scores = cv_results['mean_test_score'][matching_points]
    best_mean_score = np.max(matching_points_scores)
    best_score_idx = matching_points[np.argmax(matching_points_scores)]
    best_score_params = cv_results['params'][best_score_idx]
    return best_mean_score, best_score_params


def calc_binary_classification_tpr_fpr(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    tp = conf_mat[1, 1]
    fp = conf_mat[0, 1]
    fn = conf_mat[1, 0]
    tn = conf_mat[0, 0]
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return tpr, fpr, conf_mat


def compute_inter_class_distances(data, num_samples_per_class):
    """
    Compute all pairwise inter-class Euclidean distances between samples in a numpy array.

    Parameters:
    - data: 2D numpy array where samples of different classes are arranged sequentially.
    - num_samples_per_class: number of samples per class.

    Returns:
    - inter_class_distances - vector of size 0.5*N*(N-1)*num_samples_per_class distances. The distances are from each point in the data
    to all other points not in the same class
    """
    num_samples = data.shape[0]
    num_classes = num_samples // num_samples_per_class
    inter_class_distances = []


    # Iterate over all unique pairs of classes
    for i in range(num_classes):
        class_i = data[i * num_samples_per_class:(i + 1) * num_samples_per_class, :]
        for j in range(i + 1, num_classes):
            class_j = data[j * num_samples_per_class:(j + 1) * num_samples_per_class, :]
            distances = cdist(class_i, class_j).flatten()  # Compute pairwise distances between class i and class j
            inter_class_distances.append(distances)
    inter_class_distances = np.concatenate(inter_class_distances, axis=0)

    return inter_class_distances
