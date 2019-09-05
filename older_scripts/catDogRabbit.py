import skimage.io as io
import keras.backend as K
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import to_categorical

#import other stuff

import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io

# Get data directory
image_dir = os.path.join("./", "AnimalData")

# Get list of files
files = os.listdir(image_dir)
if ".DS_Store" in files:
    files.remove('.DS_Store')

# Dictoinary for one-hot-encoding
order = {"cat": 0, "dog": 1, "rabbit": 2}
num_classes = 3

# Create training data
xs = []
ys = []

for file in files:
    # load image
    fname = os.path.join(image_dir, file)
    img = io.imread(fname)
    # Get label from file name
    label = file.split(".")[0][0:-1]
    # Create one-hot-encoding
    y = to_categorical(order[label], num_classes)

    # Scale image between 0 and 1 - type went from unit8 to float32
    x = img.astype("float32") / 255.

    # Add to lists
    xs.append(x)
    ys.append(y)


# See one hot encoding
ys[0]

# Create X_train and y_train dataframes
X_train = np.stack(xs)
y_train = np.stack(ys)

X_train.shape, y_train.shape
K.clear_session()

model = Sequential()
model.add(Conv2D(32, input_shape=(224, 224, 3), kernel_size=3, strides=2, activation="relu", name="Conv1"))
model.add(Conv2D(32, kernel_size=3, strides=2, activation="relu", name="Conv2"))
model.add(Conv2D(32, kernel_size=3, strides=2, activation="relu", name="Conv3"))
model.add(Conv2D(32, kernel_size=3, strides=2, activation="relu", name="Conv4"))
model.add(Conv2D(32, kernel_size=3, strides=2, activation="relu", name="Conv5"))
model.add(Flatten())
model.add(Dense(16, activation="relu", name="FC"))
model.add(Dense(3, activation="softmax", name="Classification"))
model.summary()


#Training time. First set the optimizer, learning rate etc. and then compile the model. Compile is essentially moving all the model parameters / weights to the GPU. In a CPU setting its doing something similar but where its storing the network/weights is not so obvious. The loss function for multi-class classification problems is "categorical_crossentropy", in the case of binary classification we can use "binary_crossentropy".
#We also tell it to keep track of the accuracy, because we want to know how good it is as classifying. F

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["acc"])

# Training - don't worry about validation at this point
history = model.fit(X_train, y_train, epochs=20)

#We can visualise the training process from the history object returned by the model.fit().
plt.style.use("ggplot")
fig, axes = plt.subplots(1, 2)
axes[0].plot(range(20), history.history["loss"])
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Crossentropy Loss")
axes[1].plot(range(20), history.history["acc"])
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
plt.show()