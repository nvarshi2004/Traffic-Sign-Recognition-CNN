# Traffic-Sign-Recognition-CNN
# German Traffic Sign Recognition System

This project implements a Convolutional Neural Network (CNN) to classify German traffic signs from the GTSRB (German Traffic Sign Recognition Benchmark) dataset.  It achieves high accuracy in identifying various traffic signs, and also includes a real-time detection feature using a webcam.

## Description

The goal of this project is to accurately classify traffic signs, which is a crucial component of Advanced Driver Assistance Systems (ADAS).  This system can help in applications like:

* Driver alerts
* Autonomous navigation

## Dataset

The GTSRB dataset was used for training and validation.  You can find more information about it [link to GTSRB dataset]. It contains images of 43 different classes of traffic signs. [cite: 104, 105]

## Model Architecture

The CNN architecture consists of the following layers:

* Convolutional layers (32, 64, 128 filters) with ReLU activation [cite: 105]
* Max Pooling layers [cite: 105]
* Flatten layer [cite: 105]
* Dense layers (256 units, 43 units) [cite: 105]
* Dropout layer (0.5) [cite: 105]

The model summary is as follows:
