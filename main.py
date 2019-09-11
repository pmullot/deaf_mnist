# https://www.kaggle.com/ardamavi/sign-language-digits-dataset/downloads/sign-language-digits-dataset.zip/2
from keras.utils.generic_utils import CustomObjectScope
import tensorflow as tf
import getDataSet as gds
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import ReLU
from keras.layers import DepthwiseConv2D
import sys
import os
from keras.datasets import cifar10
from keras.utils import np_utils


def main():
    try:
        print('trying to load model')
        if os.path.isfile('sign_language.model'):
            model = tf.keras.models.load_model('sign_language.model', custom_objects={
                'relu6': tf.nn.relu6})
            X, X_test, Y, Y_test = gds.get_dataset()
            X = tf.keras.utils.normalize(X, axis=1)
            X_test = tf.keras.utils.normalize(X_test, axis=1)
        else:  # ValueError as err:
            # print(err)
            print('no model available, get READY TO TRAIN ONE')
            X, X_test, Y, Y_test = gds.get_dataset()

            print('normalizing images')
            X = tf.keras.utils.normalize(X, axis=1)
            X_test = tf.keras.utils.normalize(X_test, axis=1)
            Y = np_utils.to_categorical(Y)
            Y_test = np_utils.to_categorical(Y_test)

            print('building model')
            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Conv2D(
                32, (3, 3), input_shape=X.shape[1:], activation=tf.nn.relu6, padding='same'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Conv2D(
                64, (3, 3), activation=tf.nn.relu6, padding='same'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Conv2D(
                128, (3, 3), padding='same', activation=tf.nn.relu6,))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dropout(0.2))

            model.add(tf.keras.layers.Dense(
                256, kernel_constraint=tf.keras.constraints.MaxNorm(3), activation=tf.nn.relu6))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Dense(
                128, kernel_constraint=tf.keras.constraints.MaxNorm(3), activation=tf.nn.relu6))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Dense(
                64, kernel_constraint=tf.keras.constraints.MaxNorm(3), activation=tf.nn.relu6))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
            print(model.summary())

            model.compile(optimizer='adam',
                          loss='categorical_crossentropy', metrics=['accuracy'])

            print('Start Training model')
            model.fit(X, Y,  epochs=35)

            print('Model trained')
            cal_loss, val_acc = model.evaluate(X_test, Y_test)
            print('accuracy of {}'.format(val_acc))

            if (val_acc >= 0.75):
                print('Accuracy is good enough. Saving model')

                model.save('sign_language.model')
                print("Model saved to disk")
    except Exception as e:
        print(e)
        return
    print("Let's see if I can guess...")
    print()
    print()
    print()
    print()
    images = gds.loadLiveImages()
    images_norm = tf.keras.utils.normalize(images, axis=1)
    precictions = model.predict(images_norm)
    print(f'This sign represents the number {np.argmax(precictions[0])}')
    print()
    print()
    # plt.imshow(images[0], cmap=plt.cm.binary)
    # plt.show()


def load_model(file_name):
    json_file = open(file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("model_pretrained.h5")
    return load_model


main()
