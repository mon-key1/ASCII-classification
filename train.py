import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LambdaCallback
import losses_graph

path_to_mnist = 'data/mnist_custom.npz'
with np.load(path_to_mnist) as data:
    train_images, train_labels = data['x_train'], data['y_train']
    test_images, test_labels = data['x_test'], data['y_test']

print("Train images shape:", train_images.shape)
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape((45012, 28 * 28))
test_images = test_images.reshape((4092, 28 * 28))

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(256, activation='relu', input_shape=(28 * 28,)))
#model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(27, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Функция для записи loss в файл
def save_loss(epoch, logs):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    with open('res/test_loss_log.txt', 'a') as f:
        f.write(f"{test_loss}\n")
    with open('res/test_acc_log.txt', 'a') as f:
        f.write(f"{test_acc}\n")

# Callback для записи loss после каждой эпохи
loss_callback = LambdaCallback(on_epoch_end=save_loss)

history = model.fit(train_images, train_labels, epochs=130, batch_size=128, callbacks=[loss_callback])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

model.save('models/mnist_model.keras')

predictions = model.predict(test_images)
print(np.argmax(predictions[0]))

# Построение графика loss
losses_graph.create('res/test_loss_log.txt','loss')
losses_graph.create('res/test_acc_log.txt','accuracy')
