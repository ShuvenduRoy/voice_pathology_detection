import keras
import os
from sklearn.model_selection import train_test_split
from data_processing import get_labels
from keras.utils import to_categorical
import numpy as np

DATA_PATH = 'data'


def get_train_test(split_ratio=0.9, random_state=13):
    # get available class labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(os.path.join(DATA_PATH, labels[0]))
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(os.path.join(DATA_PATH, labels[0]))
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)


train_x, test_x, train_y, test_y = get_train_test()

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)


# add depth dimension for cnn
x, y, z = train_x.shape
train_x = train_x.reshape(x, y, z, 1)
test_x = test_x.reshape(test_x.shape[0], y, z, 1)

train_y_categorical = to_categorical(train_y)
test_y_categorical = to_categorical(test_y)

print(train_x.shape)
print(test_x.shape)
print(train_y_categorical.shape)
print(test_y_categorical.shape)


# build a keras model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(train_y_categorical.shape[1], activation='softmax'))

model.summary()


# train the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['categorical_accuracy'])
history = model.fit(train_x, train_y_categorical, batch_size=100, epochs=200, validation_data=(test_x, test_y_categorical))