import itertools
import os
import warnings

import cv2
from sklearn.metrics import confusion_matrix

from classes.datasets import load_fer2013

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Conv2D, \
    MaxPooling2D

from classes.evaluation_plot import evaluation_line_plot

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class EmotionClassifier:
    IMAGE_WIDTH = 48
    IMAGE_HEIGHT = 48
    COLOR_CHANNELS = 3

    FILTERS = 32
    CLASSES = 5  # 0=Angry, 1=Happy, 2=Sad, 3=Surprise, 4=Neutral

    BATCH_SIZE = 64
    EPOCHS = 200

    def __init__(self, weights=None):
        """Initializes the model. If weights is None, the weights are initialized randomly.

        Arguments:
            weights: File path where the weights are specified.
        """
        self.model = self._build_model(self.FILTERS, self.CLASSES)
        if weights is not None:
            self.model.load_weights(weights)

    def _build_model(self, filters, classes):
        """Build the model.

        Arguments:
            filters: Number of filters used in the first convolution block.
            classes: Number of classes for the final predictions.

        Returns:
            Numpy array(s) of predictions.
        """
        input_shape = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.COLOR_CHANNELS)

        model = Sequential()

        # 1st convolution block
        model.add(Conv2D(filters, kernel_size=(3, 3), activation='relu', input_shape=input_shape,
                         data_format='channels_last', kernel_regularizer=l2(0.01)))
        model.add(Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        # 2nd convolution block
        model.add(Conv2D(2 * filters, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(2 * filters, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        # 3rd convolution block
        model.add(Conv2D(4 * filters, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(4 * filters, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        # 4th convolution block
        model.add(Conv2D(8 * filters, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(8 * filters, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())

        # classification block
        model.add(Dense(8 * filters, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(4 * filters, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(2 * filters, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(classes, activation='softmax'))

        return model

    def train(self, show_evaluation=True):
        """Trains the model on the FER2013 training set.

        Arguments:
            show_evaluation: Whether to show the evaluation plot of training and test accuracy/loss.
        """

        # import training data
        x_train, y_train, x_valid, y_valid = load_fer2013()

        # augment training data
        gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True)

        train_generator = gen.flow(x_train, y_train, batch_size=self.BATCH_SIZE)
        valid_generator = gen.flow(x_valid, y_valid, batch_size=self.BATCH_SIZE)

        # compile model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # create a callback that saves the model's weights
        checkpoint_path = "cp-{epoch:04d}.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_acc',
            mode='max',
            save_best_only=True)

        # train the model
        with tf.device('/device:GPU:0'):
            history = self.model.fit(
                train_generator,
                steps_per_epoch=int(np.math.ceil(len(x_train) / self.BATCH_SIZE)),
                epochs=self.EPOCHS,
                validation_data=valid_generator,
                validation_steps=int(np.math.ceil(len(x_valid) / self.BATCH_SIZE)),
                callbacks=[cp_callback])

        # print(history.history)

        if show_evaluation:
            loss = history.history['loss']
            acc = history.history['acc']
            val_loss = history.history['val_loss']
            val_acc = history.history['val_acc']

            evaluation_line_plot(loss, x_axis_title='# Epochs', y_axis_title='Loss', x_range=range(1, len(loss) + 1))
            evaluation_line_plot(acc, x_axis_title='# Epochs', y_axis_title='Accuracy', x_range=range(1, len(loss) + 1))
            evaluation_line_plot(val_loss, x_axis_title='# Epochs', y_axis_title='Validation loss',
                                 x_range=range(1, len(loss) + 1))
            evaluation_line_plot(val_acc, x_axis_title='# Epochs', y_axis_title='Validation accuracy',
                                 x_range=range(1, len(loss) + 1))

            plt.show()

    def predict(self, image):
        """Generates output predictions for the input samples.

        Arguments:
            image: Input samples as 3-dimensional numpy array.

        Returns:
            Numpy array(s) of predictions.
        """
        # resize image to appropriate shape
        input = cv2.resize(image, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
        input = np.expand_dims(input, axis=0)

        # calculate prediction
        predictions = self.model.predict(input)
        prediction = np.argmax(predictions)

        return prediction

    def test(self):
        """Tests the model on the FER2013 test set and plots the associated confusion matrix."""

        # import training data
        x_train, y_train, x_valid, y_valid = load_fer2013()
        gen = ImageDataGenerator()
        test_generator = gen.flow(x_valid, y_valid, batch_size=self.BATCH_SIZE, shuffle=False)

        prediction = self.model.predict_generator(test_generator)

        y_valid = np.argmax(y_valid, axis=1)

        # compute confusion matrix
        cnf_matrix = confusion_matrix(y_valid, prediction.argmax(axis=1))
        np.set_printoptions(precision=2)

        # plot confusion matrix
        plt.figure()
        self._plot_confusion_matrix(cnf_matrix, classes=['angry', 'happy', 'sad', 'surprised', 'neutral'])
        plt.show()

    def _plot_confusion_matrix(self, cm, classes, cmap='Blues'):
        """Plots the normalized confusion matrix."""

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
