Aim:To use computer vision to aid agricultural robotics.
1. Identification of fruits in a tree
2. Read and understand a fruit detection algorithm using semantic segmentation.
3. Devise an algorithm for counting fruits.
4. Implement the algorithm and run on above data set.
5. Devise an algorithm to identify fresh and rotten fruit..


Detection and Classification of Apples:


We used Artificial Intelligence through the use of computer vision and deep learning
to automate apple detection, sorting and identification of rotten ones.

Step1 — Getting Training Data-
We provided sufficient sample images ( dataset ) of the apples we need the model to
detect and identify. We amended the YOLOV3 algorithm for detecting and counting
apples as well as identify rotten apples in images. We used the following dataset:
https://github.com/OlafenwaMoses/AppleDetection/releases/download/v1/apple_
detection_dataset.zip
It is divided into:
1. 563 images for training the AI model
2. 150 images for testing the trained AI model

Step2 — Training our model-
To generate our model, we trained a YOLOv3 model using ImageAI.
YOLO V3 is designed to be a multi-scaled detector, we also need features from multiple
scales.
The code to train the model is given in "detection_model_train.ipynb" file.
  
Step 3 — Predicting using our trained Model-
As we can see, the trained apple detection model is able to detect all the apples in
the image as well as identify the apple with a defect (apple with a spot). You can use
this trained model to count apples.
The code used to predict images can be accessed in the file "detection_and_classification.ipynb"

 ![alt text](https://github.com/Tajamul21/Detection-Classification-and-Semantic_Segmentation-of-apples/blob/main/Predicted%20images(Detection%20and%20Classification)/sample5.jpg)

Semantic Segmentation:

The goal of semantic image segmentation is to label each pixel of an image with a
corresponding class of what is being represented. Because we’re predicting for every
pixel in the image, this task is commonly referred to as dense prediction.
Unlike the previous tasks, the expected output in semantic segmentation are not just
labels and bounding box parameters. The output itself is a high resolution image (typically of the same size as input image) in which each pixel is classified to a particular
class. Thus it is a pixel level image classification.
The data set used for Segmentation can be found here “https://arxiv.org/abs/1909.06441”
The model for Semantic Segmentation of Apple is given in the file “Semantic segmentation.ipynb”

![alt text](https://github.com/Tajamul21/Detection-Classification-and-Semantic_Segmentation-of-apples/blob/main/Predicted%20Segmented%20Images/sample2.png)

U-Net:
U-Net is an architecture for semantic segmentation. It consists of a contracting path and an expansive path. The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels. Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU.

Training: 
Model is compiled with Adam optimizer and we use binary cross entropy loss function since there are only two classes (apple and no apple). We use Keras callbacks to implement: Learning rate decay if the validation loss does not improve for 5 continues epochs. Early stopping if the validation loss does not improve for 10 continues epochs. Save the weights only if there is improvement in validation loss. We used a batch size of 32.
