# euro-coin-classification
This is a simple project of euro coins classification using CNN model. All code is in python, for image processing we used OpenCV library and Tensorflow.

Training of CNN model is done in train_cnn_model.py file. For training we used euro coin dataset that I downloaded from Github: 
  - Pitrified/coin-dataset: https://github.com/Pitrified/coin-dataset
Before training CNN model, we crop out coins from original dataset and split the dataset to train and test data. Each class (value of coin) has 900 train samples and 100 test samples.  Modified dataset is also included in this repository (in "data" folder).

After training, we used trained model to classify coins on few sample images that we with made mobile phone. Algorithm works in two stages. In first stage we find coins in input image using OpenCV, then in second stage we use our trained model to classify coins. 

This image shows training results for out CNN model:
![model2](https://user-images.githubusercontent.com/54812954/111203316-28883a00-85c5-11eb-9d84-6a489631a744.png)

Results of algorithm on some test images:

![kovanci7_detect](https://user-images.githubusercontent.com/54812954/111203792-b8c67f00-85c5-11eb-8467-da64a95c00f2.jpg)

![kovanci6_detect](https://user-images.githubusercontent.com/54812954/111204094-11961780-85c6-11eb-8d90-c671eaec5ab8.jpg)

![kovanci9_detect](https://user-images.githubusercontent.com/54812954/111204104-14910800-85c6-11eb-8f01-45426745c382.jpg)


