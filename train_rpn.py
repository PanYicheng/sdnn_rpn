import numpy as np
import os
import keras
from keras.layers import Input, Conv2D
from keras.models import Model

path_train_data = ''

k = 9
feature_size = (16, 16, 10)

x_train = np.random.rand(100, 16, 16, 10)
class_labels = np.random.rand(100, 16, 16, k)
bbox_labels = np.random.rand(100, 16, 16, 4*k)

feature_input = Input(shape=feature_size, name='feature_input')
conv1 = Conv2D(512, (3,3), padding='same', name='shared_conv')(feature_input)

rpn_class = Conv2D(k, (1, 1), padding='same', name='rpn_class')(conv1)
rpn_bbox = Conv2D(4*k, (1, 1), padding='same', name='rpn_bbox')(conv1)

rpn_model = Model(inputs=feature_input, outputs=[rpn_class, rpn_bbox])
rpn_model.compile(optimizer='rmsprop', 
                  loss={'rpn_class':'binary_crossentropy', 'rpn_bbox':'binary_crossentropy'}, 
                  loss_weights={'rpn_class':1.0, 'rpn_bbox':0.2})
rpn_model.summary()
print(rpn_model.metrics_names)
rpn_model.fit(x_train, {'rpn_class':class_labels, 'rpn_bbox': bbox_labels}, epochs=1, batch_size=100, validation_split=0.1)

rpn_model.save('rpn.hdf5')





