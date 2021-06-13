from keras.layers import *
from keras.models import Model
import tensorflow.keras as tf

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

    def baseline(self, optimizer='sdg', loss='categorical_crossentropy', metrics=METRICS):
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
        
        model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
        )

        return model 
