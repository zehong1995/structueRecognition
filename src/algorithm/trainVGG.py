from keras.applications import VGG16
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Add
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import Model
import keras
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

from mylayer import mylayer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


base_model = VGG16(weights = "imagenet", include_top = False, input_shape=(124, 124, 3))

x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
output = Dense(12, activation='softmax')(x)

model = Model(inputs = base_model.input, outputs = output)

model.summary()

opt = keras.optimizers.Adam(lr=0.00001)
tb = keras.callbacks.TensorBoard(log_dir='./logs_VGG')

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

datagen_train = ImageDataGenerator(rotation_range=20,horizontal_flip=True,
                             vertical_flip=True,validation_split=0.2,
                             width_shift_range=0.2, height_shift_range=0.2)

#datagen_train = ImageDataGenerator()
datagen_train = datagen_train.flow_from_directory('../train',target_size=(124,124),shuffle=True,batch_size=32)

datagen_test = ImageDataGenerator()
datagen_test = datagen_test.flow_from_directory('../test', target_size=(124, 124), shuffle=True,batch_size=32)


hist = model.fit_generator(datagen_train, steps_per_epoch=158, epochs=250, callbacks=[tb])
loss, acc = model.evaluate_generator(datagen_test,steps=10,verbose=0)
print('loss = ', loss, 'acc = ', acc)

model.save('./vggtransfer_VGG.h5')
