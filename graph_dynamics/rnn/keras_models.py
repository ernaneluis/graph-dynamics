'''
Created on Jul 24, 2017

@author: cesar
'''


#import numpy as np
#from keras.engine import Merge
#from keras.models import Sequential
#from keras.layers import Dense
#import keras.backend as K

"""
xdim = 4
ydim = 1
gate = Sequential([Dense(2, input_dim=xdim)])
mlp1 = Sequential([Dense(1, input_dim=xdim)])
mlp2 = Sequential([Dense(1, input_dim=xdim)])


def merge_mode(branches):
    g, o1, o2 = branches
    # I'd have liked to write
    # return o1 * K.transpose(g[:, 0]) + o2 * K.transpose(g[:, 1])
    # but it doesn't work, and I don't know enough Keras to solve it
    return K.transpose(K.transpose(o1) * g[:, 0] + K.transpose(o2) * g[:, 1])


model = Sequential()
model.add(Merge([gate, mlp1, mlp2], output_shape=(ydim,), mode=merge_mode))
model.compile(optimizer='Adam', loss='mean_squared_error')

train_size = 19
nb_inputs = 3  # one input tensor for each branch (g, o1, o2)
x_train = [np.random.random((train_size, xdim)) for _ in range(nb_inputs)]
y_train = np.random.random((train_size, ydim))
model.fit(x_train, y_train)

def me_loss(y_true, y_pred):
    g = gate.layers[-1].output
    o1 = mlp1.layers[-1].output
    o2 = mlp2.layers[-1].output
    A = g[:, 0] * K.transpose(K.exp(-0.5 * K.square(y_true - o1)))
    B = g[:, 1] * K.transpose(K.exp(-0.5 * K.square(y_true - o2)))
    return -K.log(K.sum(A+B))

# [...] edit the compile line from above example
model.compile(optimizer='Adam', loss=me_loss)
"""