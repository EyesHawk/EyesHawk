#
# 1. mnist dataset 을 다운로드하는 모듈 작성 : module.mnist_dataset
# 2. keras 를 이용하여 간단하게 mnist 맞추는 예제 작성. (mlp 네트워크 - 렐루, 소프트맥스, 카테고리컬 크로스엔트로피, 아담)
#
# - 일단 들어본 적 있는거 같은 모듈은 최대한 다 import 해봄.
# - 서적 [모두의 딥러닝 - 조태호] 참고함
# - model 저장을 위해 h5py 모듈 install 해야 함.
#

import tensorflow as tf
import module.mnist_dataset as md
import matplotlib.pyplot as plot
# import panda as pd
import os
import numpy

from keras import Sequential, Model, Input, backend
from keras.models import load_model
from keras.layers.core import Dense, Reshape
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.activations import relu, softplus
from keras.layers.advanced_activations import LeakyReLU, PReLU, ThresholdedReLU, Softmax
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error, mean_squared_logarithmic_error
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils import np_utils


BATCH_SIZE = 100
NUM_OF_CLASSES = 10
EPOCHS = 10
LOAD_HDF5_FILE = 'resource/mnist_model/04-0.0719.hdf5'


mnist_datasets = md.get_mnist_datasets()
x_train = mnist_datasets[0][0]
y_train = mnist_datasets[0][1]
x_test = mnist_datasets[1][0]
y_test = mnist_datasets[1][1]
# x_train = mnist_datasets[0].reshape(60000, 784)
if len(x_train) == len(y_train) and len(x_test) == len(y_test):
    print("train set : " + str(len(x_train)) + " / test set : " + str(len(x_test)))
else:
    raise IOError("dataset count error")


plot.imshow(x_train[0])  # plot.imshow(x_train[0], cmap='Greys')
plot.show()

num_of_training_data = x_train.shape[0]
num_of_test_data = x_test.shape[0]
num_of_pixel = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(num_of_training_data, num_of_pixel).astype('float64') / 255
x_test = x_test.reshape(num_of_test_data, num_of_pixel).astype('float64') / 255

print(y_train[0])
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print(y_train[0])

print("[" + LOAD_HDF5_FILE + "] 을 불러옵니다.")
if os.path.exists(LOAD_HDF5_FILE):
    print("[" + LOAD_HDF5_FILE + "] 로드에 성공")
    model = load_model(LOAD_HDF5_FILE)
else:
    print("[" + LOAD_HDF5_FILE + "] 가 존재하지 않음. 처음부터 시작합니다")
    model = Sequential()
    model.add(Dense(512, input_dim=num_of_pixel, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

cur_dir = os.path.dirname(__file__)
model_path = cur_dir+'/resource/mnist_model/{epoch:02d}-{val_loss:.4f}.hdf5'
check_pointer_callback = ModelCheckpoint(filepath=model_path, save_best_only=True)
early_stopping_callback = EarlyStopping()

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[early_stopping_callback, check_pointer_callback])


score = model.evaluate(x_test, y_test)
yv_accuracy = score[1]  # 테스트셋
print('test set accuracy : %.4f' % yv_accuracy)

y_loss = history.history['loss']  # 학습셋
yv_loss = history.history['val_loss']  # 테스트셋

x_len = numpy.arange(len(y_loss))
plot.plot(x_len, y_loss, marker='.', c='blue', label='train_set_loss')
plot.plot(x_len, yv_loss, marker='.', c='red', label='test_set_loss')

plot.legend(loc='upper right')
plot.grid()
plot.xlabel('epoch')
plot.ylabel('loss')
plot.show()
