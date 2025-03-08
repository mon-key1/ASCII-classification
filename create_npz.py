import numpy as np
from PIL import Image
import os

train_folder = 'train'
test_folder = 'test'

def load_images_and_labels(folder):
    images = []
    labels = []
    
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            image_path = os.path.join(folder, filename)
            image = Image.open(image_path).convert('L')
            image_array = np.array(image)

            if filename.split('_')[-1].split('.')[0] == "a":
                label = 1
            else:
                label = 2

            images.append(image_array)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

x_train, y_train = load_images_and_labels(train_folder)
x_test, y_test = load_images_and_labels(test_folder)

np.savez('mnist_custom.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

print("Data successfully saved to mnist_custom.npz")
