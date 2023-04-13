import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import Adam, SGD

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# TASK 1
def read_file(path: Path):
    ret = []
    with open(path.as_posix(), 'r') as f:
        lines = f.read().split('\n')

    for line in lines:
        if line:
            ret.append(float(line))
    
    return np.asarray(ret)


def roc(gt_path: Path, output_path: Path):
    gt = read_file(gt_path).astype(np.uint8)
    gt = np.ones_like(gt, dtype=np.uint8) - gt
    pred = read_file(output_path)
    fp_rates = []  # x
    tp_rates = []  # y 
    step = (abs(min(pred)) + abs(max(pred)) / 1000)

    for th in np.arange(min(pred) - step, max(pred) + step, step):
        pred_th = (pred > th).astype(np.uint8)
        tp = np.sum(np.logical_and(pred_th == 1, gt == 1))
        tn = np.sum(np.logical_and(pred_th == 0, gt == 0))
        fp = np.sum(np.logical_and(pred_th == 1, gt == 0))
        fn = np.sum(np.logical_and(pred_th == 0, gt == 1))
        tpr = tp / (tp + fn)  # recall
        fpr = fp / (fp + tn)  # 1 - precision
        fp_rates.append(fpr)
        tp_rates.append(tpr)

    plt.figure(), plt.plot(fp_rates, tp_rates), plt.title('ROC Curve'), plt.show()

# TASK 2-8
def read_mnist_data(noise_factor: float = 0.2):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Add noise
    train_images_noisy = train_images + noise_factor * tf.random.normal(shape=train_images.shape) 
    test_images_noisy = test_images + noise_factor * tf.random.normal(shape=test_images.shape) 
    train_images_noisy = tf.clip_by_value(train_images_noisy, clip_value_min=0., clip_value_max=1.)
    test_images_noisy = tf.clip_by_value(test_images_noisy, clip_value_min=0., clip_value_max=1.)

    # One-hot encode labels
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    return train_images_noisy, train_images, train_labels, test_images_noisy, test_images, test_labels


def define_CNN_classifier():
    inp = Input(shape=(28, 28, 1))
    model = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')(inp)
    model = BatchNormalization()(model)
    model = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D((2, 2))(model)
    model = Dropout(0.25)(model)

    model = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(model)
    model = BatchNormalization()(model)
    model = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D((2, 2))(model)
    model = Dropout(0.25)(model)

    model = Flatten()(model)
    model = Dense(512, activation='relu', kernel_initializer='he_uniform')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.50)(model)
    model = Dense(10, activation='softmax')(model)
    
    opt = SGD(learning_rate=0.01)
    classifier = Model(inp, model)
    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier


def define_CNN_denoiser():
    inp = Input(shape=(28, 28, 1))
    # Encoder
    model = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    model = MaxPooling2D((2, 2), padding='same')(model)
    model = Conv2D(32, (3, 3), activation='relu', padding='same')(model)
    model = MaxPooling2D((2, 2), padding='same')(model)
    # Decoder
    model = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(model)
    model = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(model)
    model = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(model)
    # Autoencoder
    denoiser = Model(inp, model)
    opt = Adam(learning_rate=0.01)
    denoiser.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    return denoiser


def plot_noisy_and_denoised(noisy_images, denoised_images, n):
    plt.figure()
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.title("Noisy images")
        plt.imshow(tf.squeeze(noisy_images[i]), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        bx = plt.subplot(2, n, i + n + 1)
        plt.title("Denoised images")
        plt.imshow(tf.squeeze(denoised_images[i]), cmap='gray')
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)
    plt.show()


def mnist_classification():
    # Task 2
    noise_factor = 0.2
    (
        train_images_noisy,
        train_images,  # Clean images
        train_labels,
        test_images_noisy,
        test_images,  # Clean images
        test_labels
    ) = read_mnist_data(noise_factor)
    
    # Task 3
    model_path = 'ex4/task3_model.h5'
    if Path(model_path).exists():
        classifier = tf.keras.models.load_model(model_path)
        print('Task 3 model loaded')
    else:
        classifier = define_CNN_classifier()
        classifier.fit(train_images, train_labels, epochs=10, batch_size=32)
        classifier.save(model_path)
    
    # Task 4
    _, test_acc = classifier.evaluate(test_images, test_labels)
    print(f'Classification accuracy for clean test images: {test_acc:.3f}')

    # Task 5
    _, test_acc = classifier.evaluate(test_images_noisy, test_labels)
    print(f'Classification accuracy for noisy test images: {test_acc:.3f}')

    # Task 6
    model_path = 'ex4/task6_model.h5'
    if Path(model_path).exists():
        denoiser = tf.keras.models.load_model(model_path)
        print('Task 6 model loaded')
    else:
        denoiser = define_CNN_denoiser()
        denoiser.fit(train_images_noisy, train_images, epochs=10, batch_size=32)
        denoiser.save(model_path)

    test_images_denoised = denoiser(test_images_noisy).numpy()
    plot_noisy_and_denoised(test_images_noisy, test_images_denoised, 10)

    # Task 7
    _, test_acc = classifier.evaluate(test_images_denoised, test_labels)
    print(f'Classification accuracy for denoised test images: {test_acc:.3f}')

    # Task 8
    model_path = 'ex4/task8_model.h5'
    if Path(model_path).exists():
        classifier_noisy = tf.keras.models.load_model(model_path)
        print('Task 8 model loaded')
    else:
        classifier_noisy = define_CNN_classifier()
        classifier_noisy.fit(train_images_noisy, train_labels, epochs=10, batch_size=32)
        classifier_noisy.save(model_path)
    
    _, test_acc = classifier_noisy.evaluate(test_images_noisy, test_labels)
    print(f'Classification accuracy for noisy test images (trained with noisy images): {test_acc:.3f}')


def main():
    do_task1 = True
    do_task2to8 = True
    # Task 1
    if do_task1:
        gt_fn = 'ex4/detector_groundtruth.dat'
        output_fn = 'ex4/detector_output.dat'

        gt_path = Path(gt_fn)
        output_path = Path(output_fn)

        roc(gt_path, output_path)

    # Task 2-8
    if do_task2to8:
        mnist_classification()


if __name__ == '__main__':
    main()
