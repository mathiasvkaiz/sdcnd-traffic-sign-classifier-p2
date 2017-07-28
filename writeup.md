#**Traffic Sign Recognition** 

##Writeup

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

[image1]: ./examples/histogram.jpg "Visualization"
[image2]: ./examples/original.jpg "Original"
[image2_1]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/normalize.jpg "Normalized"
[image4]: ./images/img1.jpg "Traffic Sign 1"
[image5]: ./images/img2.jpg "Traffic Sign 2"
[image6]: ./images/img3.jpg "Traffic Sign 3"
[image7]: ./images/img4.jpg "Traffic Sign 4"
[image8]: ./images/img5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/mathiasvkaiz/sdcnd-traffic-sign-classifier-p2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing how the data is distributed. You can see for each class the frequency (or count of imges) that are related to each class.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


As a first step, I decided to convert the images to grayscale because having a grayscaled image it is easier for the network to detect edges and patterns as the light differences are easier to detect. This gray scaling step gave me following example output.

![alt text][image2] ![alt text][image2_1]


As second step i decided to normalize all images according to the calculation "pixels / 255" (Max - Min -> 255 - 0). The purpose of normalization is to have a the image in a more familiar range to get consistency and reducing high diffenrences in image. This normalization step gave me following example output.

![alt text][image3]



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		     | 32x32x1 Grayscale image   							           | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Dropout     	        | Probability 0.9                            	|
| Leaky RELU	          |	Max (x, alpha=0.1 * x)                      |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x6  				          |
| Convolution 5x5	     | 1x1 stride, valid padding, outputs 10x10x16 |
| Dropout     	        | Probability 0.9                            	|
| Leaky RELU	          |	Max (x, alpha=0.1 * x)                      |
| Max pooling	      	  | 2x2 stride,  outputs 5x5x16   				          |
| Fully connected		    | input 5*5*16, output flat 400               |
| Dropout     	        | Probability 0.9                            	|
| Leaky RELU	          | Max (x, alpha=0.1 * x)                      |
| Fully connected		    | input 400, output flat 120                  |
| Dropout     	        | Probability 0.9                            	|
| Leaky RELU	          | Max (x, alpha=0.1 * x)                      |
| Fully connected		    | input 120, output flat 84                   |
| Dropout     	        | Probability 0.9                            	|
| Leaky RELU	          | Max (x, alpha=0.1 * x)                      |
| Fully connected		    | input 84 , output flat 10                   |
| Dropout     	        | Probability 0.9                            	|
| Leaky RELU	          | Max (x, alpha=0.1 * x)                      |
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following parameters:
* Learning rate at 0.002 
* beta1 for decaying learning rate at 0.5
* keep probability for dropout layer at 0.9
* Epochs 20
* Batch size at 32

A too small learning rate can lead to a network not learning in appropriate speed. A learning rate too high can lead to a bad accuracy as the risk not to descent but to hop around or even ascent is increasing the higher the rate. 

Beta is the exponential decay rate for the 1st moment in the optimizer and 0.5 is a good value to start with.

Keep probability is used for dropout layer defining the probability that nodes are kept and not thrown away. This technique leads to a more robust network as dropping out randmoly nodes decrease e.g. overfitting risk.

20 epochs with decreasing learning rate led to a desired accuracy result.

Batch size of 32 is often a good candidate. Also increases memory  usage as matrix multiplications increase. Small bacth sizes have more noise in error calculation, this noise is often helpful to avoid local minimas.


As optimizer i used AdamOptimizer with decaying learning rate which lead to good (not perfect) results.



####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I took the known LeNet and added iteratively process adding preprocessing and tuning hyper parameters. I also added dropout layers for more robust network. We can see that the training results are pretty good at nearly 100%. Validation accuracy is on 0.935 percent meaning that we have no overfitting here as a low validation accuracy would indicate overfitting compared to a high train accuracy. The train and validation accuracy is calculated in the train phase (17th cell in the notebook).

Also the test validation seems quite reasonable. The differences can be explained by different numbers of training exmaples for each class so it is not said that we have a euqal distibuted example count for every class. This could be done better in preprocessing step but was not regarded here.  The test accuracy was calculated (18th cell in the notebook) at last as we should do test validation only once we have finished training and validatiing the model. Otherwise we would incorporate the test examples into the model.

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.935
* test set accuracy of 0.918

If an iterative approach based on LeNet was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I started with the notbook from Lesson having the LedNet. But as already described i was not able to increase accuracy over 0.89. So i played around with normalization values. Is started with normalization step as suggested in the notebook but discovered that i significantly got better results with the (pixels / 255) approach coming to .90. After that i added standard distributed weights and biases in a small range coming around .91. That also increased accuracy as the weight and bias initializing is a very important topic. The distributed range shouldn't be too large otherwise the accuracy drops out to .5 and below.  

I added droput layers before  all RelU units to get a more robuts network and avid overfitting.
Also i changed the actiation function from ReLU to Leaky ReLU as a ReLU can die meaning that it outputs alwys the same output 0. Leaky ReLU has a chance to recover that unit.

I also added decaying learning rate to avoid lower accuracy on later epochs. I examined that with the decaying learning rate and on 20 epochs minimized the risk of overfitting as whe weights are not updated that "heavily" anymore at a later stage of training.



###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first, third and fifth images should be much more easy to predcit as, beside the different angles, we have a clear pattern of what these images are meant for.

The second image could be hard to predict as we have much noise around it and the important part is only a smal area on that image. 

The fourth image might be difficult to classify because we have a different angle on the sign. The crossing line is not horizontally but diagonal so this could be ard to recognize the pattern.




####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution      		| General caution					 				|
| Speed limit (30km/h)					| Yield											|
| Speed limit (50km/h)     			| Speed limit (50km/h) 										|
| No entry      		| Keep left   									| 
| Right-of-way at the next intersection			| Right-of-way at the next intersection      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares not favorably to the accuracy on the test set of as there we had nearly 91%. As mentioned above, those images that are not preprocessed in an optimal way were predicted wrong. So we can see how important preporcessing step is. On the other hand, in a real life scenario we cannot assume, that we have always the best view on a traffic sign so other techniques need to be applied as well (beside the preprocessing) to get good results in real life scenario as well.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 73th cell of the Ipython notebook.


For the first image, the model is absolute sure that this is a General caution sign (probability of 1), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| General caution   									| 
| .00     				| Right-of-way at the next intersection 										|
| .00					| Pedestrians									|
| .00	      			| 			Traffic signals 				|
| .00				    | Double curve      							|



For the second image, the model is relative sure that this is a Yield sign (probability of 0.67), but the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .67         			| Yield   									| 
| .20     				| Right-of-way at the next intersection 										|
| .05					| Keep left									|
| .01	      			| 			Roundabout mandatory	 				|
| .01				    | Road work      							|


For the third image, the model is relative sure that this is a Speed limit (50km/h) sign (probability of 0.55), and the image does contain a Speed limit (50km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .55         			| Speed limit (50km/h)   									| 
| .44     				| Speed limit (30km/h) 										|
| .00					| Speed limit (80km/h)										|
| .00	      			| 			Wild animals crossing		 				|
| .00				    | Stop      							|



For the fourth image, the model is very sure that this is a Keep Left sign (probability of 0.99), but the image does contain a No entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Keep left   									| 
| .003     				| Yield 										|
| .00					| Turn right ahead											|
| .00	      			| 			Right-of-way at the next intersection		 				|
| .00				    | Children crossing      							|



For the fifth image, the model is absolute sure that this is a Right-of-way at the next intersection sign (probability of 1), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Right-of-way at the next intersection   									| 
| .00     				| General caution 										|
| .00					| Double curve									|
| .00	      			| 			Pedestrians 				|
| .00				    | Dangerous curve to the left      							|




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


