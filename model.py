import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

path_to_mnist = 'mnist_custom.npz'
with np.load(path_to_mnist) as data:
    train_images, train_labels = data['x_train'], data['y_train']
    test_images, test_labels = data['x_test'], data['y_test']

print("Train images shape:", train_images.shape)
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape((248, 28 * 28))
test_images = test_images.reshape((27, 28 * 28))

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=60, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

model.save('mnist_model.keras')

predictions = model.predict(test_images)
print(np.argmax(predictions[0]))
