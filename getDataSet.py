# Arda Mavi
import os
import numpy as np
from os import listdir
from imageio import imread
from PIL import Image

# from scipy import imread, imresize
from tensorflow.keras.utils import to_categorical

# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Settings:
img_size = 64
grayscale_images = True
num_class = 10
test_size = 0.2


def get_img(data_path):
    # Getting image array from path:
    # img = imread(data_path, flatten=grayscale_images)
    img = Image.open(data_path).convert('L')
    img = img.resize((img_size, img_size))
    # img = imresize(img, (img_size, img_size, 1 if grayscale_images else 3))
    img = np.array(img)
    return img


def get_dataset(dataset_path='Dataset'):
    # Getting all data from data path:
    try:
        X = np.load('npy_dataset/X.npy')
        Y = np.load('npy_dataset/Y.npy')
    except:
        labels = listdir(dataset_path)  # Geting labels
        X = []
        Y = []
        for i, label in enumerate(labels):
            datas_path = dataset_path+'/'+label
            print('loading images for number {}'.format(label))
            for data in listdir(datas_path):
                img = get_img(datas_path+'/'+data)
                X.append(img)
                Y.append(int(label))
        # Create dateset:
        #X = 1-np.array(X).astype('float32')/255.
        Y = np.array(Y).astype('int64')
        # Y = to_categorical(Y, num_class)
        if not os.path.exists('npy_dataset/'):
            os.makedirs('npy_dataset/')
        np.save('npy_dataset/X.npy', X)
        np.save('npy_dataset/Y.npy', Y)
    X, X_test, Y, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42)

    return X, X_test, Y, Y_test


def loadLiveImages(dataset_path='Live_Examples'):
    imgs = []
    for data in listdir(dataset_path):
        imgs.append(get_img(dataset_path+'/'+data))
    return imgs


if __name__ == '__main__':
    get_dataset()
