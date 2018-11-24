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

order = {
    "cat" : 0,
    "dog" : 1,
    "rabbit" : 2
}
x = "cat"
y = np.array([1, 0, 0])

to_categorical(order["cat"], num_classes=3)
to_categorical(order["rabbit"], num_classes=3)
to_categorical(order["dog"], num_classes=3)
fname = "cat1.jpg"
#x = io.imread(fname)
label = fname.split(".")[0][0:-1]
y = to_categorical(order[label], num_classes=3);
y