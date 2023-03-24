import tensorflow as tf
from tensorflow import keras
from keras import layers, utils
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from pathlib import Path
import matplotlib.pyplot as plt

PATH_TO_DATASET_FOLDER = 'ex2/GTSRB_subset_2/'

def read_data(data_path: Path, extension: str = 'jpg') -> np.ndarray:
    if data_path.exists:
        image_data = []
        for fn in data_path.rglob('*.' + extension):
            image_data.append(np.asarray(Image.open(fn)))
        return np.asarray(image_data)
    else:
        raise FileNotFoundError

if __name__ == '__main__':
    # START OF TASK 2
    path_to_dataset = Path(PATH_TO_DATASET_FOLDER)
    class1_path = path_to_dataset / 'class1'
    class2_path = path_to_dataset / 'class2'

    # Read all the data
    X0 = read_data(class1_path)
    y0 = np.zeros(X0.shape[0], dtype=np.uint8)
    X1 = read_data(class2_path)
    y1 = np.ones(X1.shape[0], dtype=np.uint8)

    # Concatenate data before split
    X = np.concatenate([X0, X1])
    y = np.concatenate([y0, y1])
    X, y = shuffle(X, y)
    print('X-shape:', X.shape, 'y-shape:', y.shape)

    # Shuffle, split and modify
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    y_train = utils.np_utils.to_categorical(y_train, num_classes=2)
    # y_test = utils.np_utils.to_categorical(y_test, num_classes=2)
    X_train = X_train.astype('float64') / 255.0
    X_test = X_test.astype('float64') / 255.0
    print(
        'X_train-shape:', X_train.shape,
        'y_train-shape:', y_train.shape,
        '& X_test-shape:', X_test.shape,
        'y_test-shape:', y_test.shape
        )
    # END OF TASK 2

    # START OF TASK 3
    model = keras.Sequential(
        [
            layers.Flatten(input_shape=X_train[0].shape),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(2, activation='softmax')
        ]
    )
    model.summary()
    # END OF TASK 3

    # START OF TASK 4
    e = 10
    model.compile(optimizer='SGD', loss='categorical_crossentropy')
    history = model.fit(X_train, y_train, epochs=e)
    plt.figure(), plt.plot(history.history['loss']), plt.show()

    predicted_labels = model.predict(X_test) 
    predicted_labels = np.argmax(predicted_labels, axis=1)
    test_acc = 1 - np.count_nonzero(y_test - predicted_labels) / len(predicted_labels)
    print('\nTest accuracy:', test_acc)
    # END OF TASK 4