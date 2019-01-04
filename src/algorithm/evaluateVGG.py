from keras.applications import VGG16
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Add
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import Model
from mylayer import mylayer
import keras
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


model = keras.models.load_model('./vggtransfer_VGG.h5', custom_objects={'mylayer': mylayer})
model.load_weights('./vggtransfer_VGG.h5')

model.summary()

datagen_train = ImageDataGenerator(rotation_range=20,horizontal_flip=True,
                             vertical_flip=True,validation_split=0.2,
                             width_shift_range=0.2, height_shift_range=0.2)

#datagen_train = ImageDataGenerator()
datagen_train = datagen_train.flow_from_directory('../train',target_size=(124,124),shuffle=True,batch_size=32)

datagen_test = ImageDataGenerator()
datagen_test = datagen_test.flow_from_directory('../test', target_size=(124, 124), shuffle=False,batch_size=32)



#hist = model.fit_generator(datagen_train, steps_per_epoch=158, epochs=50, callbacks=[tb])
import time
start_time = time.time()
loss, acc = model.evaluate_generator(datagen_test,steps=38,verbose=0)
end_time = time.time()
fps = 38*32 / (end_time - start_time)
print('loss = ', loss, 'acc = ', acc)
print(fps)
