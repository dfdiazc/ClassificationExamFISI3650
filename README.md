# Classification Exam FISI-3650

This repository contains a solution to the classification exam for the entrepreneurship course FISI-3650 at the University of the Andes.

## Objetives and Dataset

The project aims to obtain a model, a convolutional neural network in this case, that can accurately classify different types of rice based on an image. To do this, the provided dataset contains 50000 images of 5 different varieties of rice: arborio, basmati, ipsala, jasmine, and karacadag. An additional 10000 images are provided for validation. The model will be implemented in Python, using the **Keras** API from the **Tensorflow** library.

## The model

Initially, the dataset is loaded, using 80% of the images for training, while the remaining 20% will be used for validation. The images are loaded into the model in batches of 64, and were resized to 50x50 pixels in order to optimize CPU and GPU usage. Then the model itself is defined. The first layer rescales the images and normalizes the RGB values. Then, three convolution layers are defined, each with a corresponding pooling layer and ReLU activation functions. A dropout layer is also defined in order to improve the quality of the model. Finally, two dense, fully connected layers are defined. The first one is a 256 unit layer with a ReLU activation function, same as before. The second one is the final output layer, a 5 unit one with a Softmax activation function.

Having defined the model, the Adam optimizer is chosen and a Cross Entropy function is defined for the loss function. Now, the model is trained for 10 epochs and saved in a .h5 file.

## Results

Using the obtained model it is possile now to run the *eval.py* file, obtaining a 96.82% accuracy for the **Test** data.
