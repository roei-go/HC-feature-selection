from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def multiple_classifiers_fit_predict(classifiers, X_train, y_train, X_test, y_test, score_func=accuracy_score):
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
    for i in range(len(classifiers)):
        # in case the classifier has a feature selector, set the selected features attribute to None to make sure
        # the feature selector is re-fitted to the data
        if hasattr(classifiers[i], 'feature_selector'):
            classifiers[i].feature_selector.selected_features = None
        classifiers[i].fit(X_train, y_train)
        if hasattr(classifiers[i], 'feature_selector'):
            num_features[i] = classifiers[i].feature_selector.selected_features.shape[0]
        y_pred = classifiers[i].predict(X_test)
        scores[i] = score_func(y_test, y_pred)

    return  scores, num_features


def preprocessing_model(input_shape, preprocess_layer, noise_model=None):

    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_layer(inputs)
    if noise_model is not None:
        outputs = noise_model(x)
    else:
        outputs = x
    return tf.keras.Model(inputs, outputs)


def get_feature_extractor(base_model,input_shape, pooling_layer):
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
    X = np.concatenate(X,axis=0)
    return X


def get_images_from_supervised_set(ds,label_set,img_size=[224,224], max_samples_per_class=100):
    images = []
    labels = []
    samples_per_class = {label : 0 for label in label_set}
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