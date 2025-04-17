# Traffic-Sign-Recognition-CNN
# German Traffic Sign Recognition System

This project implements a Convolutional Neural Network (CNN) to classify German traffic signs from the GTSRB (German Traffic Sign Recognition Benchmark) dataset.  It achieves high accuracy in identifying various traffic signs, and also includes a real-time detection feature using a webcam.

## Description

The goal of this project is to accurately classify traffic signs, which is a crucial component of Advanced Driver Assistance Systems (ADAS).  This system can help in applications like:

* Driver alerts
* Autonomous navigation

## Dataset

The GTSRB dataset was used for training and validation.  You can find more information about it "https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign". It contains images of 43 different classes of traffic signs. 
## Requirements

* Python 3.7+
* TensorFlow
* Keras
* OpenCV (cv2)
* NumPy
* Pandas
* Matplotlib

## Model Architecture

The CNN architecture consists of the following layers:

* Convolutional layers (32, 64, 128 filters) with ReLU activation 
* Max Pooling layers 
* Flatten layer 
* Dense layers (256 units, 43 units) 
* Dropout layer (0.5) 

The model summary is as follows:

# Traffic Sign Recognition System

This project implements a Convolutional Neural Network (CNN) to classify German traffic signs from the GTSRB (German Traffic Sign Recognition Benchmark) dataset. It achieves high accuracy in identifying various traffic signs and also includes a real-time detection feature using a webcam.

## Description

The goal of this project is to accurately classify traffic signs, which is a crucial component of Advanced Driver Assistance Systems (ADAS). This system can help in applications like:

* Driver alerts
* Autonomous navigation

## Dataset

The GTSRB dataset was used for training and validation. It contains images of 43 different classes of traffic signs.

## Model Architecture

The CNN architecture consists of the following layers:

* Convolutional layers (32, 64, 128 filters) with ReLU activation
* Max Pooling layers
* Flatten layer
* Dense layers (256 units, 43 units)
* Dropout layer (0.5)

The model summary is as follows:

Model: "sequential" Layer (type) Output Shape Param #
conv2d (Conv2D) (None, 30, 30, 32) 4896
max_pooling2d (MaxPooling2D) (None, 15, 15, 32) 0
conv2d_1 (Conv2D) (None, 13, 13, 64) 18496
max_pooling2d_1 (MaxPooling2D) (None, 6, 6, 64) 0
conv2d_2 (Conv2D) (None, 4, 4, 128) 73856
max_pooling2d_2 (MaxPooling2D) (None, 2, 2, 128) 0
flatten (Flatten) (None, 512) 0
dense (Dense) (None, 256) 131328
dropout (Dropout) (None, 256) 0
dense_1 (Dense) (None, 43) 11051
Total params:                235627 (920.42 KB)
Trainable params:            235627 (920.42 KB)
Non-trainable params:        0
## Results

The model achieved a validation accuracy of 99.53% on the validation set.

