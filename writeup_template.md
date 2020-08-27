# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./my_images/original_images_with_class.png "Visualization with classes"
[image2]: ./my_images/three_histos.png "3 histograms"
[image3]: ./my_images/grayscale.png "Grayscale image"
[image4]: ./my_images/translation.png "Translated image"
[image5]: ./my_images/noise_image.png "Noise image"
[image6]: ./my_images/augmented_histo.png "Augmented histogram"
[image7]: ./my_images/lenet.png "LeNet"
[image8]: ./my_images/epoch_result.png "Epoch Result"
[image9]: ./my_images/loss_function.png "Loss function"
[image10]: ./traffic_signs/vz_1.png "Sign 1"
[image11]: ./traffic_signs/vz_2.png "Sign 2"
[image12]: ./traffic_signs/vz_3.png "Sign 3"
[image13]: ./traffic_signs/vz_4.png "Sign 4"
[image14]: ./traffic_signs/vz_5.png "Sign 5"
[image15]: ./traffic_signs/vz_6.png "Sign 6"
[image16]: ./my_images/six_grayscale.png "6 input images"
[image17]: ./my_images/prediction.png "Prediction"
[image18]: ./my_images/prediction_2.png "Prediction of all images with 3 candidates"
[image19]: ./my_images/softmax.png "Softmax"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/raffael87/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of the original training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x1
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
First step is to generate randomly indexes and access the training and plot the image with its class.
This can then be cross checked with the csv file where to each id a description is held.
![alt text][image1]

But this is not enough. In order to know the distribution of the classes and to get an impression about the data set, we have to create histograms for it.
![alt text][image2]
The first histogram shows the distribution for the training classes. In order to have a good learning result it is essential, that the data is distributed more evenly. Several classes have fewer (min is 120 and max around 2000) data samples.
As a result the model could be biased towards the classes with more data samples. A good approach is to augment the data with generated / modified images from the same class.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to gray-scale because as we saw in the lab from this degree, gray-scale images work really good with the LeNet and the minst data set.
Traffic signs are more shape and number / text oriented and not that much color.
The gray-scale images got also normalized to the range of -1 and 1. As suggested in the code section the operation normalized = (train - 128) / 128 is used. The mean in the end was -0.36 which is a good value, close to zero.

![alt text][image3]

As already mentioned I decided to generate additional data to get an even distributed training set.
With the original training set, the accuracy of the network was around 0.90 which was not a good value.

I created methods to scale the image and to add some noise to them. Unfortunately, the time to create the data was really high.
That's why I changed the augmentation. Instead I added images from the training set to the classes.
Instead of 34799 images we have now 86430 images. Each class has 2010 images. This is not optimal, I would have liked to use the other methods, but is is better then an uneven distribution.

Here is an example of an original image and a translated image by 2px:
![alt text][image4]

Here is an example of an original image and a the noise pixels:
![alt text][image5]

After augmenting, the histogram is as follows (it has a peak which i guess is an implementation error):
![alt text][image6]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
As a starting point I used the model from the lab:
![alt text][image7]

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 normalized grayscale image 					 |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x6 	 |
| RELU					    |												                       |
| Max pooling       | 2x2 stride, valid padding, outputs = 14x14x6 |
| Convolution 5x5	  | 1x1 stride, valid padding, outputs 28x28x6   |
| RELU					    |												                       |
| Max pooling       | 2x2 stride, valid padding, outputs = 5x5x16  |
| Flatten           | outputs = 400                                |
| Fully connected		| input = 400, outputs = 120       						 |
| RELU					    |												                       |
| Dropout				    | keep_prob = 0.7	                             |
| Fully connected		| input = 120, outputs = 84       						 |
| RELU					    |												                       |
| Dropout				    | keep_prob = 0.7	                             |
| Fully connected		| input = 84, outputs = 43       							 |

After several tests, this was the table with good results. I added two dropout layers with a higher keep probability so that the network is forced to learn loosing connections.
I tested also adding less drop outs or putting them in different places. But in the end the best result was between relu and fully connected.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer which we were already using throughout the course.
The final settings for the hyperparameters were:
training rate: 0.0005
epochs: 30
bathc size: 128
mu: 0
sigma: 0.1

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My approach consisted of both, trial and error and using a know architecture.
As we already used a architecture which seemed to fit for also classifying traffic signs I started with the LeNet from the lab.
I did not modify the training set.
So the first step was to have a running training pipeline. But the results were not good. Running multiple times, the accuracy never made it over 0.90.
The first impression out of this was then to modify the hyperparameters. But this approach was kind of random. Some times I got values between .90 and .91 and on the next change I got .85
Then I looked into the data and because in the lab we were using gray-scale images, I implemented a method to change the input data into gray-scale and normalize it. The improvement was a bit better, but still not good enough.
Looking again into the data I saw how badly distributed it was. So the next steps were to implement a solution to augment the data set and make it more balanced.
After this the results got much better and a accuracy of greater than .93 was achieved in multiple runs.

I went through the courses in order to see how the parameters can be chosen properly. One point was to keep all the time some results from previous runs and also visualize the data.
Often an over fitting happened after running too many epochs. Another good approach was to chose a small learning rate. So the increment is not that high but it learns better.

Another big boost came when adding the drop out layers. Even though now a days there are more advanced techniques it worked quite well. I have chosen a more conservative keep value of .7

![alt text][image8]
![alt text][image9]


My final model results were:
* validation set accuracy of 0.956
* test set accuracy of 0.940

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image10] ![alt text][image11] ![alt text][image12]
![alt text][image13] ![alt text][image14] ![alt text][image15]

I have chosen images which were having the same geometrical form, but also images with border and where the sign maybe overlapps a bit. In general I would think that nearly similar signs would have problems.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Images after gray-scale and normalizing
![alt text][image16]

Prediction vectors. 4 out of 6 images were classified correctly and a Test Accuracy of 0.667 is reached.
![alt text][image17]

If we look at the prediction, we can take a look at the top 5, but also at the best three candidates and show the image and its prediction.
![alt text][image18]

What surprises me is that it was not possible to classify the stop sign.

In general the accuracy is not as good as expected.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a construction sign (probability of 1), and the image does contain a stop sign. So all others have 0 probability.
For the second  image, the model is relatively sure that this is a kids running sign (probability of 1), and the image does contain a kids running sign. So all others have 0 probability.
For the third, the model is relatively sure that this is a ! sign (probability of .98), but the image does not contain a ! sign. Moreover also in the candidates there is no stop sign. So this is a true false positive.
For the fourth, also a miss classification, the model is quite sure yet, wit .78. But also in the other two candidates there is no animal crossing sign. An interessting fact is, that a round sign has a probability of .22
Fifth and sixth image are classified with 1.0 probability.
![alt text][image19].
