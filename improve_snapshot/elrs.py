import numpy as np
import tensorflow as tf
import keras
class EarlyLRFindStopping(keras.callbacks.Callback):

    def __init__(self, schedule=None, patience=0):
        super(EarlyLRFindStopping, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.schedule = schedule

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf

    def on_epoch_begin(self, epoch, logs=None):
            if not hasattr(self.model.optimizer, "lr"):
                raise ValueError('Optimizer must have a "lr" attribute.')
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            scheduled_lr = self.schedule(epoch, lr)
            tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
            print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))