from keras.layers import *
from keras.models import Model
from keras.models import Sequential
import tensorflow as tf

METRICS = [
    tf.metrics.BinaryAccuracy(name='acc')]

class Baseline():
    def __init__(self,x_in, x_out):
        self.x_in = x_in
        self.x_out = x_out

    def get_model(self, init='glorot_uniform'):
        model = Sequential()
        model.add(Dense(64, input_dim=self.x_in, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(64, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(self.x_out, kernel_initializer=init, activation=tf.nn.sigmoid))
        
        # compile model
        loss = ['binary_crossentropy', 'categorical_crossentropy'][self.x_out > 2]
        
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        model.compile(loss=loss, 
                    optimizer=optimizer, 
                    metrics=METRICS)
        return model