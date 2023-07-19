from mlsocket import MLSocket
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import socket
HOST = socket.gethostname()
PORT = 5001
import keras


from keras.datasets import mnist     # MNIST dataset is included in Keras
from keras.models import Sequential  # Model type to be used

from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils     
import socket

import tensorflow as tf

ip_address = socket.gethostbyname(socket.gethostname())


def do_nothing():
    return 0

print('Hii')
# Make a keras model

weights1 = [0,0,0,0,0,0]
weights2 = [0,0,0,0,0,0]
arch1 = 0
arch2 = 0

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



model1 = make_model()
model2 = make_model()

# Send data
with MLSocket() as s1, MLSocket() as s2:

    s1.connect((HOST, PORT)) # Connect to the port and host
    s2.connect((HOST, 5000)) # Connect to the port and host
    
    data1 = s1.recv(1024)
    if not data1:
        s1.close()
    print(data1)
    data2 = s2.recv(1024)
    if not data2:
        s2.close()
    print(data2)
    while(True):
        
        a = input('''
        1. Train Models
        2. Receive Models
        3. Save Models
        4. Ensemble Models
                ''')
        s1.send(str.encode(a))
        s2.send(str.encode(a))
        if(a == 'Train Models'):
            data1 = s1.recv(1024)
            data2 = s2.recv(1024)
            if not data1 and not data2:
                break
            print(data1)
            print(data2)
            i = 0
            while(data1 != bytes(f'{ip_address} -> Models Trained Successfully' , 'utf-8')  and 
                    data2 != bytes(f'{ip_address} -> Models Trained Successfully' , 'utf-8')):
                do_nothing()
                data1 = s1.recv(1024)
                data2 = s2.recv(1024)
            print(data1)
            print(data2)
        elif(a == 'Save Models'):
            data1 = s1.recv(1024)
            data2 = s2.recv(1024)
            if not data1 and not data2:
                break
            print(data1)
            print(data2)
            while(data1 != bytes(f'{ip_address} -> Models Saving Completed' , 'utf-8')  and 
                    data2 != bytes(f'{ip_address} -> Models Saving Completed' , 'utf-8')):
                do_nothing()
                data1 = s1.recv(1024)
                data2 = s2.recv(1024)
            print(data1)
            print(data2)

        elif(a == 'Receive Models'):


            data1 = s1.recv(1024)
            data2 = s2.recv(1024)
            if not data1 and not data2:
                break
            print(data1)
            print(data2)
            i = 0
            j = 0
            while(data1 != bytes(f'{ip_address} -> Models Sending Completed' , 'utf-8')  and 
                  data2 != bytes(f'{ip_address} -> Models Sending Completed' , 'utf-8')):

                data1 = s1.recv(1024)
                if(isinstance(data1,np.ndarray)):
                    weights1[i] = data1
                    print(weights1[i].shape)
                    i +=1
                
                data2 = s2.recv(1024)
                if(isinstance(data1,np.ndarray)):
                    weights2[j] = data2
                    print(weights2[j].shape)
                    j +=1

            model1.set_weights(weights1)
            model2.set_weights(weights2)

            model1.save('Model1.h5')
            model2.save('Model2.h5')    


            

            print(data1)
            print(data2)

        elif(a == 'Ensemble Models'):

            model1 = keras.models.load_model('Model1.h5')
            model2 = keras.models.load_model('Model2.h5')

            models = [model1 , model2]
            model_input = tf.keras.Input(shape=(784,))
            model_outputs = [model(model_input) for model in models]
            ensemble_output = tf.keras.layers.Average()(model_outputs)
            ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output) 
            ensemble_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            ensemble_model.fit(X_train, Y_train,
                               batch_size=128, epochs=5,
                               verbose=1)
            a = ensemble_model.evaluate(X_test,Y_test)
            print(a)

        else:
            break




        

    s1.close()
    s2.close()