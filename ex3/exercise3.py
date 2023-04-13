from tensorflow import keras
from keras import utils
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


if __name__ == "__main__":
    # START OF TASK 2
    model = keras.Sequential(
        [
            keras.Input((64, 64, 3)),
            keras.layers.Conv2D(10, 3, strides=(2, 2), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(10, 3, strides=(2, 2), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(2, activation='softmax')
        ]
    )
    model.build()
    model.summary()
    # END OF TASK 2

    # START OF TASK 3
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
    y_test = utils.np_utils.to_categorical(y_test, num_classes=2)
    X_train = X_train.astype('float64') / 255.0
    X_test = X_test.astype('float64') / 255.0

    # Train and test
    e = 20
    model.compile(optimizer='SGD', loss=keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=e, batch_size=32, validation_data=(X_test, y_test))
    plt.figure(), plt.plot(history.history['loss']), plt.show()

    test_loss, test_acc = model.evaluate(X_test,  y_test)
    print('\nTest accuracy:', test_acc)

