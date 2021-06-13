from keras.layers import *
from keras.models import Model

class Baseline():
    def __init__(self,x_in, x_out):
        self.x_in = x_in
        self.x_out = x_out

    def baseline(self, optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy']):
        
        x = Conv2D(32, (3,3), activation='relu', padding='same', use_bias=False)(self.x_in)
        x = Conv2D(32, (3,3), activation='relu', padding='same', use_bias=False)(x)
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