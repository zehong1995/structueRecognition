from keras import backend as K
import keras
from keras.engine.topology import Layer

class mylayer(Layer):
    def __init__(self, **kwargs):
        super(mylayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name = 'kernel',
                                     shape=(1,),dtype='float32',trainable=True,initializer='uniform')
        print(self.kernel.shape)
        #self.gamma = K.variable(1.0, dtype="float32")
        #self.trainable_weights = [self.gamma]
        super(mylayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.kernel * inputs