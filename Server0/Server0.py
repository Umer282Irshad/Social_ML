import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines

from keras.datasets import mnist     # MNIST dataset is included in Keras
from keras.models import Sequential  # Model type to be used

from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils     
import socket

from mlsocket import MLSocket

import tensorflow as tf

import keras

hostname = socket.gethostname()

port = 5000
_id = 0



(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)


X_train = X_train.reshape(60000, 784) # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
X_test = X_test.reshape(10000, 784)   # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.

X_train = X_train.astype('float32')   # change integers to 32-bit floating point numbers
X_test = X_test.astype('float32')

X_train /= 255                        # normalize each value for each pixel for the entire vector for each input
X_test /= 255


print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

nb_classes = 10 # number of unique digits

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

def make_model():

    model = Sequential()                                 # Linear stacking of layers

    # Convolution Layer 1
    model.add(Dense(512, input_shape=(784,))) #(784,) is not a typo -- that represents a 784 length vector!
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_model(myModel):
    myModel.fit(X_train, Y_train,
          batch_size=128, epochs=5,
          verbose=1)
    return myModel


def save_model(myModel):
       myModel.save(f'Model{_id}.h5')




def send_model(conn , myModel):
        weights = myModel.get_weights()
        w1 = np.array(weights[0])
        w2 = np.array(weights[1])
        w3 = np.array(weights[2])
        w4 = np.array(weights[3])
        w5 = np.array(weights[4])
        w6 = np.array(weights[5])

        weights = np.array([w1,w2,w3,w4,w5,w6],dtype = object)
        for i in range(6):
             print(weights[i].shape)
             conn.send(weights[i])


        
def server_program():

    with MLSocket() as s:
        s.bind((hostname, port))
        s.listen()
        conn, address = s.accept()
        print("Connection from: " + str(address))
        with conn:
            
            conn.send(bytes(f'{str(address)} -> Connection Made' , 'utf-8'))
            model = make_model()
            while(True):
                data = conn.recv(1024)
                if(data == b'Train Models'):
                    print(data)
                    print('Model Training Started')
                    conn.send(bytes(f'{str(address[0])} -> Models Training Started' , 'utf-8'))
                    model = train_model(make_model())
                    conn.send(bytes(f'{str(address[0])} -> Models Trained Successfully' , 'utf-8'))
                    print('Model Trained Successfully')
                elif(data == b'Save Models'):
                    print(data)
                    print('Models Saving Started')
                    conn.send(bytes(f'{str(address[0])} -> Models Saving Started' , 'utf-8'))
                    save_model(model)
                    conn.send(bytes(f'{str(address[0])} -> Models Saving Completed' , 'utf-8'))
                    print('Models Saving Completed')
                elif(data == b'Receive Models'):
                    print(data)
                    print('Models Sending Started')
                    conn.send(bytes(f'{str(address[0])} -> Models Sending Started' , 'utf-8'))
                    model = keras.models.load_model(f'Model{_id}.h5')
                    print('Model Loaded')
                    send_model(conn,model)
                    conn.send(bytes(f'{str(address[0])} -> Models Sending Completed' , 'utf-8'))
                    print('Models Sending Completed')
                     

if __name__ == '__main__':
    server_program()