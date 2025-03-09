import numpy as np
from PIL import Image
import os

train_folder = 'data/data_f/train'
test_folder = 'data/data_f/test'

def load_images_and_labels(folder):
    images = []
    labels = []
    
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            image_path = os.path.join(folder, filename)
            image = Image.open(image_path).convert('L')
            image_array = np.array(image)

            label = 0

            if filename[0] == 'H':
                label = 1
            elif filename[0] == 'E':
                label = 2
            elif filename[0] == 'L':
                label = 3
            elif filename[0] == 'O':
                label = 4

            images.append(image_array)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    return images, labels

x_train, y_train = load_images_and_labels(train_folder)
x_test, y_test = load_images_and_labels(test_folder)

np.savez('data/mnist_custom.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
