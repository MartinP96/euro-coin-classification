# euro-coin-classification
This is a simple project of euro coins classification using CNN model. All code is in python, for image processing we used OpenCV library and Tensorflow.

Training of CNN model is done in train_cnn_model.py file. For training we used euro coin dataset that I downloaded from Github: 
  - Pitrified/coin-dataset: https://github.com/Pitrified/coin-dataset
Before training CNN model, we crop out coins from original dataset and split the dataset to train and test data. Each class (value of coin) has 900 train samples and 100 test samples.  Modified dataset is also included in this repository (in "data" folder).

After training, we used trained model to classify coins on few sample images that we with made mobile phone. Algorithm works in two stages. In first stage we find coins in input image using OpenCV, then in second stage we use our trained model to classify coins. 

