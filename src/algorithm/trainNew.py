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

feature_conv_6 = base_model.output
f_6 = Flatten()(feature_conv_6)
#f_6 = mylayer()(f_6)

length = 4608

#feature_1 = base_model.get_layer(name='block1_conv2').output
feature_2 = base_model.get_layer(name='block2_conv2').output
feature_3 = base_model.get_layer(name='block3_conv3').output
feature_4 = base_model.get_layer(name='block4_conv3').output
feature_5 = base_model.get_layer(name='block5_conv3').output

#feature_conv_1 = Conv2D(1, (1, 1), padding='same')(feature_1)
#feature_conv_1 = Flatten()(feature_conv_1)
#feature_conv_dense_1 = Dense(25088, activation='relu')(feature_conv_1)
#f_1 = mylayer()(feature_conv_dense_1)

feature_conv_2 = Conv2D(1, (1, 1), padding='same')(feature_2)
feature_conv_2 = Flatten()(feature_conv_2)
feature_conv_dense_2 = Dense(length, activation='relu')(feature_conv_2)
f_2 = mylayer()(feature_conv_dense_2)

feature_conv_3 = Conv2D(1, (1, 1), padding='same')(feature_3)
feature_conv_3 = Flatten()(feature_conv_3)
feature_conv_dense_3 = Dense(length, activation='relu')(feature_conv_3)
f_3 = mylayer()(feature_conv_dense_3)

feature_conv_4 = Conv2D(1, (1, 1), padding='same')(feature_4)
feature_conv_4 = Flatten()(feature_conv_4)
feature_conv_dense_4 = Dense(length, activation='relu')(feature_conv_4)
f_4 = mylayer()(feature_conv_dense_4)

feature_conv_5 = Conv2D(1, (1, 1), padding='same')(feature_5)
feature_conv_5 = Flatten()(feature_conv_5)
feature_conv_dense_5 = Dense(length, activation='relu')(feature_conv_5)
f_5 = mylayer()(feature_conv_dense_5)



f_merged = Add()([f_2, f_3, f_4, f_5, f_6])

x = Dense(1024, activation="relu")(f_merged)
x = Dense(512, activation="relu")(x)
output = Dense(12, activation="softmax")(x)

model = Model(inputs = base_model.input, outputs = output)

model.summary()


opt = keras.optimizers.Adam(lr=0.00001)
tb = keras.callbacks.TensorBoard(log_dir='./logs')

#Firstly, train the Dense-layer
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

datagen_train = ImageDataGenerator(rotation_range=20,horizontal_flip=True,
                             vertical_flip=True,validation_split=0.2,
                             width_shift_range=0.2, height_shift_range=0.2)

#datagen_train = ImageDataGenerator()
datagen_train = datagen_train.flow_from_directory('../train',target_size=(124,124),shuffle=True,batch_size=32)

datagen_test = ImageDataGenerator()
datagen_test = datagen_test.flow_from_directory('../test', target_size=(124, 124), shuffle=True,batch_size=32)


hist = model.fit_generator(datagen_train, steps_per_epoch=158, epochs=50, callbacks=[tb])
loss, acc = model.evaluate_generator(datagen_test,steps=10,verbose=0)
print('loss = ', loss, 'acc = ', acc)


for layer in base_model.layers:
    layer.trainable = True

opt = keras.optimizers.Adam(lr=0.00001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(datagen_train, steps_per_epoch=316, epochs=50, callbacks=[tb])
loss, acc = model.evaluate_generator(datagen_test,steps=10,verbose=0)
print('loss = ', loss, 'acc = ', acc)

hist = model.fit_generator(datagen_train, steps_per_epoch=316, epochs=50, callbacks=[tb])
loss, acc = model.evaluate_generator(datagen_test,steps=10,verbose=0)
print('loss = ', loss, 'acc = ', acc)

hist = model.fit_generator(datagen_train, steps_per_epoch=316, epochs=100, callbacks=[tb])
loss, acc = model.evaluate_generator(datagen_test,steps=10,verbose=0)
print('loss = ', loss, 'acc = ', acc)

model.save('./vggtransfer_last.h5')
