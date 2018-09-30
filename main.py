import argparse
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
from sklearn.model_selection import train_test_split
from data_processing import get_labels
from keras.utils import to_categorical
import numpy as np

DATA_PATH = 'data'
parser = argparse.ArgumentParser(description='Train a neural machine translation model')

parser.add_argument('--model', default='cnn', help='Model type: cnn/gru')
parser.add_argument('--log', default='default', help='Tensorboard log dir')
parser.add_argument('--n_epochs', default=300, help='number of epoch')
parser.add_argument('--lr', default=1e-4, type=float, help='number of epoch')
parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')

# Parse arguments
args = parser.parse_args()


def get_train_test(split_ratio=0.9, random_state=13):
    # get available class labels
    labels, indices, _ = get_labels()

    # Getting first arrays
    X = np.load('data\\' + labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load('data\\' + label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)


train_x, test_x, train_y, test_y = get_train_test()

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

train_y_categorical = to_categorical(train_y)
test_y_categorical = to_categorical(test_y)

if args.model == 'cnn':
    # add depth dimension for cnn
    x, y, z = train_x.shape
    train_x = train_x.reshape(x, y, z, 1)
    test_x = test_x.reshape(test_x.shape[0], y, z, 1)

    print(train_x.shape)
    print(test_x.shape)
    print(train_y_categorical.shape)
    print(test_y_categorical.shape)

    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(y, z, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))

    model.summary()
    # train the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(args.lr),
                  metrics=['categorical_accuracy'])
    history = model.fit(train_x, train_y_categorical, batch_size=128, epochs=350,
                        callbacks=[keras.callbacks.TensorBoard(log_dir='logs\\'+args.log, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)],
                        validation_data=(test_x, test_y_categorical))

if args.model == 'gru':
    model = Sequential()

    model.add(layers.GRU(128, activation='relu', input_shape=(None, train_x.shape[2])))
    model.add(layers.Dense(2, activation='softmax'))

    model.summary()

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=5e-4),
                  metrics=['categorical_accuracy'])
    history = model.fit(train_x, train_y_categorical, batch_size=100, epochs=250,
                        callbacks=[
                            keras.callbacks.TensorBoard(log_dir='logs\\' + args.log, histogram_freq=0, batch_size=32,
                                                        write_graph=True, write_grads=False, write_images=False,
                                                        embeddings_freq=0, embeddings_layer_names=None,
                                                        embeddings_metadata=None, embeddings_data=None)],
                        validation_data=(test_x, test_y_categorical))
