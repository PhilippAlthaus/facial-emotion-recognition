import numpy
import pandas
import tensorflow


def load_fer2013():
    train_df, eval_df = _load_fer()

    x_train, y_train = _preprocess_fer(train_df)
    x_valid, y_valid = _preprocess_fer(eval_df)

    return x_train, y_train, x_valid, y_valid


def _load_fer():
    # load FER2013 dataset
    df = pandas.read_csv('datasets/fer2013.csv', sep=',')

    # df = df.drop(df[df.emotion == 1].index)
    # df = df.drop(df[df.emotion == 2].index)

    # df.loc[df['emotion'] > 1, ['emotion']] = df['emotion'] - 2

    # split training and evaluation data
    train_df = df[df['Usage'] == 'Training']
    eval_df = df[df['Usage'] == 'PublicTest']

    return train_df, eval_df


def _preprocess_fer(df, label_col='emotion', feature_col='pixels'):
    labels, features = df.loc[:, label_col].values.astype(numpy.int32), [numpy.fromstring(image, numpy.float32, sep=' ')
                                                                         for image in df.loc[:, feature_col].values]

    labels = [tensorflow.keras.utils.to_categorical(l, num_classes=5) for l in labels]

    features = numpy.stack((features,) * 3, axis=-1)
    features /= 255
    features = features.reshape((features.shape[0], 48, 48, 3))

    return features, labels
