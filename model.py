import csv
import os
import numpy as np
from scipy.misc import imread
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D


# Note that I use Keras2 for this project
DATADIR = 'data/'
IMGDIR = os.path.join(DATADIR, 'IMG/')
CSVPATH = os.path.join(DATADIR, 'driving_log.csv')
CORRECTION_ANGLE = 0.065  # decided after small exeriments


# Read the driving_log.csv file
# Header:
# [center_img, left_img, right_img, steering angle, throttle, brake, speed]
def read_csv(csvfile):
    with open(csvfile) as ifp:
        reader = csv.reader(ifp)
        csvlines = [line for line in reader]
        # Remove the header line
        return csvlines[1:]


def ceildiv(a, b):
    return -(-a // b)


# Generator for fit data
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        # zipped shuffling by sklearn.utils.shuffle
        shuffle(samples, random_state=0)

        # Loop over batches of lines read in from driving_log.csv
        for offset in range(0, num_samples, batch_size):
            samples_batch = samples[offset:offset+batch_size]
            images_batch = []
            angles_batch = []
            for sample in samples_batch:
                # Read images as RGB
                path_center = os.path.join(IMGDIR, os.path.basename(sample[0]))
                path_left = os.path.join(IMGDIR, os.path.basename(sample[1]))
                path_right = os.path.join(IMGDIR, os.path.basename(sample[2]))
                # RGB form is used in drive.py:model.predict()
                image_center = imread(path_center, mode='RGB')
                image_left = imread(path_left, mode='RGB')
                image_right = imread(path_right, mode='RGB')
                # Augment images with a left-right flipped version.
                image_flipped = np.copy(np.fliplr(image_center))
                images_batch.append(image_center)
                images_batch.append(image_left)
                images_batch.append(image_right)
                images_batch.append(image_flipped)

                # Correct angles to the driving angle.
                angle_center = float(sample[3])
                angle_left = angle_center + CORRECTION_ANGLE
                angle_right = angle_center - CORRECTION_ANGLE
                angle_flipped = -angle_center
                angles_batch.append(angle_center)
                angles_batch.append(angle_left)
                angles_batch.append(angle_right)
                angles_batch.append(angle_flipped)

            # Return a batch of size 4*batch_size
            X_batch = np.array(images_batch)
            y_batch = np.array(angles_batch)

            yield shuffle(X_batch, y_batch, random_state=0)


if __name__ == '__main__':
    lines = read_csv(CSVPATH)
    # Split lines of driving_log.csv into training and validation in 80:20
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    # Get #of times the generators will be called in each epoch.
    train_steps = ceildiv(len(train_samples), 32)
    validation_steps = ceildiv(len(validation_samples), 32)
    # Get generators for training and validation data
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # Define CNN Model
    model = Sequential()
    # Crop the background of the car and the lane
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    # Normalize to [-1, 1]
    model.add(Lambda(lambda x: x/255. - 0.5))
    # Convolution Layers
    model.add(Conv2D(24, 5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, 5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, 3, strides=(1, 1), activation='relu'))
    model.add(Conv2D(64, 3, strides=(1, 1), activation='relu'))
    model.add(Flatten())
    # Fully connected layers
    model.add(Dense(100))
    #model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    # Optimizer
    model.compile(loss='mse', optimizer='adam')

    # Use fit_generator (for Keras>2.0) with batch generators
    model.fit_generator(train_generator, train_steps, epochs=5, \
                        validation_data=validation_generator, \
                        validation_steps=validation_steps, \
                        max_queue_size=10, workers=1, initial_epoch=0)
    # save the model
    model.save('model.h5')
