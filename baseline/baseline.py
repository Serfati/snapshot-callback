from keras.layers import *
from keras.models import Model
from keras.models import Sequential
import tensorflow as tf

METRICS = [
    tf.metrics.Accuracy(name='acc'),
    tf.metrics.TruePositives(name='tpr'),
    tf.metrics.FalsePositives(name='fpr'),
    tf.metrics.Precision(name='precision'),
    tf.metrics.AUC(name='auc-roc', curve='ROC'),
    tf.metrics.AUC(name='auc-pr', curve='PR')
]

class Baseline():
    def __init__(self,x_in, x_out):
        self.x_in = x_in
        self.x_out = x_out

    def baseline(self, optimizer='sgd', loss='categorical_crossentropy', init='glorot_uniform', metrics=METRICS):
        loss = ['binary_crossentropy', 'categorical_crossentropy'][self.x_out > 2]
        x = Conv2D(16, (2,2), activation='relu', padding='same', use_bias=False)(self.x_in)
        x = Conv2D(16, (2,2), activation='relu', padding='same', use_bias=False)(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        out = Dense(self.x_out, activation='softmax')(x)
        model = Model(inputs=self.x_in, outputs=out)
    
        # compile model 
        model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
        )

        return model 

    # let's create a function that creates the model (required for KerasClassifier) 
    # while accepting the hyperparameters we want to tune 
    # we also pass some default values such as optimizer='rmsprop'
    def create_model(self, optimizer='sgd', init='glorot_uniform'):
        model = Sequential()
        model.add(Dense(64, input_dim=self.x_in, kernel_initializer=init, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dense(self.x_out, kernel_initializer=init, activation=tf.nn.softmax))

        # compile model
        loss = ['binary_crossentropy', 'categorical_crossentropy'][self.x_out > 2]

        model.compile(loss=loss, 
                    optimizer=optimizer, 
                    metrics=METRICS)

        return model