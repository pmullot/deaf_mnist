# https://www.kaggle.com/ardamavi/sign-language-digits-dataset/downloads/sign-language-digits-dataset.zip/2
import tensorflow as tf
import getDataSet as gds
import matplotlib.pyplot as plt
import numpy as np

try:
    print('trying to load model')
    model = tf.keras.models.load_model('sign_language.model')
    X, X_test, Y, Y_test = gds.get_dataset()
    X = tf.keras.utils.normalize(X, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
except:
    print('no model available, get READY TO TRAIN ONE')
    X, X_test, Y, Y_test = gds.get_dataset()
    print('normalizing images')
    X = tf.keras.utils.normalize(X, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    print('building model')
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu6))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('Start Training model')
    model.fit(X, Y, epochs=800)

    print('Model trained')
    cal_loss, val_acc = model.evaluate(X_test, Y_test)
    print('accuracy of {}'.format(val_acc))

    if (val_acc >= 0.75):
        print('Accuracy is good enough. Saving model')
        model.save('sign_language.model')


print("Let's see if I can guess!")
images = gds.loadLiveImages()
images = tf.keras.utils.normalize(images, axis=1)
precictions = model.predict(images)
print(np.argmax(precictions[0]))

# plt.imshow(images[0], cmap=plt.cm.binary)
# plt.show()
