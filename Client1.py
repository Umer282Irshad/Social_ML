import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines

from keras.datasets import mnist     # MNIST dataset is included in Keras
from keras.models import Sequential  # Model type to be used

from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils     
import socket

from mlsocket import MLSocket

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
    mssg = f'Model{_id} Trained Successfully'
    return bytes(mssg,'utf-8')
    
def save_model(myModel):
    myModel.save(f'Model{_id}.h5')

def send_model(conn , myModel):
        conn.send(myModel)
        
def server_program():

    with MLSocket() as s:
        s.bind((hostname, port))
        s.listen()
        conn, address = s.accept()
        print("Connection from: " + str(address))
        with conn:
            # print('receiving w1')
            # w1 = conn.recv(1024) # This will block until it receives all the data send by the client, with the step size of 1024 bytes.
            # print('w1 received')
            # print('receiving w2')
            # w2 = conn.recv(1024) # This will block until it receives all the data send by the client, with the step size of 1024 bytes.
            # print('w2 received')
            # print('receiving w3')
            # w3 = conn.recv(1024) # This will block until it receives all the data send by the client, with the step size of 1024 bytes.
            # print('w3 received')
            # print('receiving w4')
            # w4 = conn.recv(1024) # This will block until it receives all the data send by the client, with the step size of 1024 bytes.
            # print('w4 received')
            # print('receiving w5')
            # w5 = conn.recv(1024) # This will block until it receives all the data send by the client, with the step size of 1024 bytes.
            # print('w5 received')
            # print('receiving w6')
            # w6 = conn.recv(1024) # This will block until it receives all the data send by the client, with the step size of 1024 bytes.
            # print('w6 received')
            # print('receiving model')
            # model = conn.recv(1024) # This will also block until it receives all the data.
            # print('model received')
            # pprint(model.to_json())
            conn.send(b'Connection Made')

            while(True):
                data = conn.recv(1024)
                if(data == b'Train Models'):
                    print(data)
                    conn.send(b'Models Training Started')
                    model = make_model()
                    mssg = train_model(model)
                    conn.send(mssg)
                else:
                    print('sad')
                    break

if __name__ == '__main__':
    server_program()