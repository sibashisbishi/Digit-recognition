import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
from matplotlib import pylab
import tensorflow as tf
import keras

#b) Let’s set a seed value, so that we can control our models randomness

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)
print(rng)
#c) The first step is to set directory paths, for safekeeping!

root_dir = os.path.abspath('../..')
print(root_dir)
data_dir = os.path.join(root_dir,'TensorInstall')
print(data_dir)
sub_dir = os.path.join(data_dir,'digitrecognition')
print(sub_dir)
# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)



train = pd.read_csv(os.path.join(sub_dir, 'train.csv'))
test = pd.read_csv(os.path.join(sub_dir, 'Test.csv'))

sample_submission = pd.read_csv(os.path.join(sub_dir, 'Sample_Submission.csv'))

print(train.head())
print(train.filename)
img_name = rng.choice(train.filename)
print(img_name)
filepath = os.path.join(sub_dir, 'train', img_name)
print(filepath)
img = imread(filepath, flatten=True)
print("Img shape:",img.shape)
pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()

temp = []
for img_name in train.filename:
    image_path = os.path.join(sub_dir, 'train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

train_x = np.stack(temp)
print(train_x.shape)
#print("Train_x",train_x)
train_x /= 255.0
#print("Train_x /255",train_x)
train_x = train_x.reshape(-1, 784).astype('float32')

temp = []
i=0
for img_name in test.filename:
    image_path = os.path.join(sub_dir, 'test', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

test_x = np.stack(temp)

test_x /= 255.0
test_x = test_x.reshape(-1, 784).astype('float32')
train_y = keras.utils.np_utils.to_categorical(train.label.values)
#print("train_y",train_y)

########################################
#                                      #
#  train_x.shape (49000, 784)          #
#  train_x.shape[0] 49000              #
#  train_x.shape[0]*0.7 34300.0        #
########################################

#take a split size of 70: 30 for train set vs validation set

split_size = int(train_x.shape[0]*0.7)

print(train_x[:split_size])


print()
print(train_x[split_size:])
train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]
train.label.ix[split_size:]

# define vars
input_num_units = 784
hidden_num_units = 50
output_num_units = 10

epochs = 7
batch_size = 128

# import keras modules

from keras.models import Sequential
from keras.layers import Dense

# create model
model = Sequential([
  Dense(output_dim=hidden_num_units, input_dim=input_num_units, activation='relu'),
  Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



trained_model = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))

pred = model.predict_classes(test_x)

img_name = rng.choice(test.filename)
filepath = os.path.join(sub_dir, 'test', img_name)

img = imread(filepath, flatten=True)

test_index = int(img_name.split('.')[0]) - train.shape[0]

print ("Prediction is: ", pred[test_index])

pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()
sample_submission.filename = test.filename; sample_submission.label = pred
sample_submission.to_csv(os.path.join(sub_dir, 'sub02.csv'), index=False)
