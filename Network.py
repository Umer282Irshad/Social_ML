import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers

from keras.datasets import mnist     # MNIST dataset is included in Keras
from keras.models import Sequential  # Model type to be used

from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils     
import socket

from mlsocket import MLSocket


_id = 1


(X_train, y_train), (X_test, y_test) = mnist.load_data()




X_train = X_train.reshape(60000, 28, 28, 1) #add an additional dimension to represent the single-channel
X_test = X_test.reshape(10000, 28, 28, 1)

X_train = X_train.astype('float32')         # change integers to 32-bit floating point numbers
X_test = X_test.astype('float32')

X_train /= 255                              # normalize each value for each pixel for the entire vector for each input
X_test /= 255



nb_classes = 10 # number of unique digits

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization

def make_model():

    model = Sequential()                                 # Linear stacking of layers

    # Convolution Layer 1
    model.add(Conv2D(32, (3, 3), input_shape=(28,28,1))) # 32 different 3x3 kernels -- so 32 feature maps
    model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
    convLayer01 = Activation('relu')                     # activation
    model.add(convLayer01)

    # Convolution Layer 2
    model.add(Conv2D(32, (3, 3)))                        # 32 different 3x3 kernels -- so 32 feature maps
    model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
    model.add(Activation('relu'))                        # activation
    convLayer02 = MaxPooling2D(pool_size=(2,2))          # Pool the max values over a 2x2 kernel
    model.add(convLayer02)

    # Convolution Layer 3
    model.add(Conv2D(64,(3, 3)))                         # 64 different 3x3 kernels -- so 64 feature maps
    model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
    convLayer03 = Activation('relu')                     # activation
    model.add(convLayer03)

    # Convolution Layer 4
    model.add(Conv2D(64, (3, 3)))                        # 64 different 3x3 kernels -- so 64 feature maps
    model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
    model.add(Activation('relu'))                        # activation
    convLayer04 = MaxPooling2D(pool_size=(2,2))          # Pool the max values over a 2x2 kernel
    model.add(convLayer04)
    model.add(Flatten())                                 # Flatten final 4x4x64 output matrix into a 1024-length vector

    # Fully Connected Layer 5
    model.add(Dense(512))                                # 512 FCN nodes
    model.add(BatchNormalization())                      # normalization
    model.add(Activation('relu'))                        # activation

    # Fully Connected Layer 6                       
    model.add(Dropout(0.2))                              # 20% dropout of randomly selected nodes
    model.add(Dense(10))                                 # final 10 FCN nodes
    model.add(Activation('softmax'))                     # softmax activation


    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_model(myModel):
    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                            height_shift_range=0.08, zoom_range=0.08)

    test_gen = ImageDataGenerator()

    train_generator = gen.flow(X_train, Y_train, batch_size=128)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=128)

    myModel.fit_generator(train_generator, steps_per_epoch=60000//128, epochs=5, verbose=1, 
                        validation_data=test_generator, validation_steps=10000//128)
    
def save_model(myModel):
    myModel.save(f'Model{_id}.h5')

def send_model(conn , myModel):
        conn.send(myModel)
        
def server_program():

    host = socket.gethostname()
    port = 5001  # initiate port no above 1024

    with MLSocket() as s:
        s.bind((host, port)) # Connect to the port and host

        s.listen(2)
    # configure how many client the server can listen simultaneously
        conn, address = s.accept()  # accept new connection
        print("Connection from: " + str(address))
        while True:
            # receive data stream. it won't accept data packet greater than 1024 bytes
            data = conn.recv(1024)
            if not data:
                # if data is not received break
                break
            print("from connected user: " + str(data))
            conn.send(data[::-1])  # send data to the client

    conn.close()  # close the connection


if __name__ == '__main__':
    server_program()