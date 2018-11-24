
# coding: utf-8

# In[1]:


import keras
from keras.models import *
from keras.layers import *
import numpy as np

x = np.array([[3]]); x.shape

x[np.newaxis, :].shape

model = Sequential()
model.add(Dense(1, input_dim=1, ))
model.summary()

model.layers[0].weights

y = np.array([[32]])

y.shape

from keras.optimizers import *

opt = Adam(lr=1)
#STG - basic - stochastic gradient descent with momentum
#with momentum - keeping the amount that you stepped in the first step into the next step.
#the gradient may change but momentum helps you to move faster towards the solution.
#ADAM may be more useful as it maximises momentum
model.compile(optimizer=opt, loss="mse")
# fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
model.fit(x, y,epochs=100)
y_pred = model.predict(x)
(y, y_pred)


weights = model.get_weights()
weights
import keras.backend as K
K.clear_session()


model = Sequential()
model.add(Dense(1, input_dim=(1)))
model.summary()
model.predict(x)

model.set_weights(weights)

model.predict(x)

::new section

weights[1]
model.layers[0].set_weights(weights)


for i, layer in enumerate(model.layers):
    if i > 50:
        layer.trainable = True
    else:
        layer.trainable = False

K.clear_session()
model = Sequential()
model.add(Dense(1, input_dim=2))

x = np.array([[3, 12]])
y = np.array([[123]])

model.summary()

opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss="mse")

x.shape
y.shape

model.fit(x, y, epochs=10)

y_pred = model.predict(x);
(y, y_pred)

model = Sequential()
model.add(Dense(2, input_dim=2))
model.summary()


# $$
# y_1 = w1 x1 + w2 x2 + b1 \\
# y_2 = w3 x1 + w4 x2 + b2
# $$

opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss="mse")

y = np.array([[32, 43]])

x.shape, y.shape
model.compile(optimizer=opt, )
model.fit(x, y, epochs=100)
y_pred = model.predict(x)
(y, y_pred)

model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Dense(1, activation="softmax"))
model.summary()

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(2, 2), strides=(2, 2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(filters=6, kernel_size=(2, 2), strides=(2, 2), padding="same"))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.summary()


# $$
# W - F + P*2 / S +1 
# $$
import skimage as sk
# SEGMENTATION
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(2, 2), strides=(2, 2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(filters=6, kernel_size=(2, 2), strides=(2, 2), padding="same"))
model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same"))
model.add(Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="softmax"))
model.add(UpSampling2D(2))
model.add(Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="softmax"))
model.add(UpSampling2D(2))
model.add(Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="softmax"))
#model.add(Dense(32, activation="relu"))
#model.add(Dense(3, activation="softmax"))
model.summary()

