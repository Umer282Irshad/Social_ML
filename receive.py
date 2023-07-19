from mlsocket import MLSocket
import socket
HOST = socket.gethostname()
PORT = 65432

from pprint import pprint

with MLSocket() as s:
    s.bind((HOST, PORT))
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
        data = conn.recv(1024)
        print(data)
        conn.send(b'Byeeeee')