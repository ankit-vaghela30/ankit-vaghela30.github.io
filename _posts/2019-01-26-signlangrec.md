---
title: "American sign language recognition"
date: 2019-01-26
tags: [Deep learning Machine learning, Data science, Tensorflow, Python]
#header:
#  image: "/images/perceptron/percept.jpg"
excerpt: "Computer vision, Deep learning"
mathjax: "true"
---

# Sign-language-recognition
This is a project developed as a final project for the course Advanced Data Analytics taught by Dr Yi Hong at University of Georgia. It was voted as **Best project in class**.

## Aim
To develop a real time sign language classification with client-server architecture where one can use the application to detect their hands and classify that hand gesture based upon the previously trained model.

## Architecture
![alt]({{ site.url }}{{ site.baseurl }}/images/Sign-Language-Prediction/architecture.png)

## Hand Detection
### Dataset
We used EgoHands dataset which is a dataset for hands in complex egocentric interactions. It contains pixel level annotations (>15000 ground truth labels) where hands are located across 4800 images. All images are captured across 48 different environments (indoor, outdoor) and activities (playing cards, chess, jenga, solving puzzles etc).

### Model
* We took an existing model (ssd_mobilenet_v1_coco) from tensorflow object detection api and retrained its final layer to detect hands.
* We saved a frozen copy of the above model
* We integrated it with tensorflow object detection API to detect the hand images.

## Gesture Classification
### Dataset
![alt]({{ site.url }}{{ site.baseurl }}/images/Sign-Language-Prediction/hands.png)

* We used ASL alphabet dataset from Kaggle which has 87000 images each with 200x200 pixels.
* 29 classes: 26 alphabets and 3 new classes for "space", "delete" and "nothing"

### model
* We used a pretrained VGG16 model which was trained on Imagenet dataset.
* We used transfer learning concept where we added a fully connected layer at the end, made top layers untrainable and then trained it on ASL alphabet dataset.
* We got 91% accuracy on test dataset.
* We saved and loaded this model as h5py file.
![alt]({{ site.url }}{{ site.baseurl }}/images/Sign-Language-Prediction/training.png)

## Project Report 
For more detailed description, Please have a look at our [project report](https://github.com/ankit-vaghela30/sign-language-recognition/blob/master/ADA_paper.pdf)

## Demo
Check our project demo [here](https://www.youtube.com/watch?v=qDAso3HYtMg&t=11s)

## References
* (https://www.kaggle.com/grassknoted/asl-alphabet/home)
* https://towardsdatascience.com/transfer-learning-946518f95666
* https://www.kaggle.com/paultimothymooney/interpret-sign-language-with-deep-learning/notebook 
