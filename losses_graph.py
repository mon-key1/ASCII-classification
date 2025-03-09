import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LambdaCallback

def create(filename,param):
    with open(filename, 'r') as f:
        losses = [float(line.strip()) for line in f]
    plt.plot(losses)
    plt.title(f'Model {param}')
    plt.ylabel(f'{param}')
    plt.xlabel('Epoch')
    plt.show()

#create("res/test_loss_log.txt",'loss')
#create("res/test_acc_log.txt",'acc')
