
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


# $$
# y = wx + b
# $$


from keras.optimizers import *


# In[49]:


opt = Adam(lr=1)


# In[50]:


model.compile(optimizer=opt, loss="mse")


# In[51]:


# fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
model.fit(x, y,epochs=100)


# In[56]:


y_pred = model.predict(x)


# In[58]:


(y, y_pred)


# In[61]:


weights = model.get_weights()


# In[62]:


weights


# In[64]:


import keras.backend as K


# In[66]:


K.clear_session()


# In[67]:


model = Sequential()
model.add(Dense(1, input_dim=(1)))
model.summary()


# In[68]:


model.predict(x)


# In[70]:


model.set_weights(weights)


# In[71]:


model.predict(x)


# In[74]:


weights[1]


# In[77]:


model.layers[0].set_weights(weights)


# In[78]:


for i, layer in enumerate(model.layers):
    if i > 50:
        layer.trainable = True
    else:
        layer.trainable = False


# In[ ]:





# In[103]:


K.clear_session()


# In[104]:


model = Sequential()
model.add(Dense(1, input_dim=2))


# In[105]:


x = np.array([[3, 12]])
y = np.array([[123]])


# In[106]:


model.summary()


# $$
# y = w_1 x_1 + w_2 x_2 + b
# $$

# In[107]:


opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss="mse")


# In[108]:


x.shape


# In[109]:


y.shape


# In[110]:


model.fit(x, y, epochs=10)


# In[112]:


y_pred = model.predict(x);
(y, y_pred)


# In[113]:


model = Sequential()
model.add(Dense(2, input_dim=2))
model.summary()


# $$
# y_1 = w1 x1 + w2 x2 + b1 \\
# y_2 = w3 x1 + w4 x2 + b2
# $$

# In[117]:


opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss="mse")


# In[121]:


y = np.array([[32, 43]])


# In[120]:


x.shape, y.shape


# In[ ]:


model.compile(optimizer=opt, )


# In[123]:


model.fit(x, y, epochs=100)


# In[124]:


y_pred = model.predict(x)
(y, y_pred)


# In[125]:


model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Dense(1, activation="softmax"))
model.summary()


# In[192]:


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

# In[205]:


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


# In[ ]:


[]

