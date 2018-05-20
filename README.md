# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/dataset_distribution.png "Data Set Distribution"
[image2]: ./img/example_img.png "What the Data Set Looks Like"
[image3]: ./img/grayscale_img.png "Grayscale"
[image4]: ./img/downloaded_img.png "Downloaded Images"

---

### Data Set Summary & Exploration

#### 1. Data Set Summary

* The size of training set:            `34799`
* The size of the validation set:      `4410`
* The size of test set:                `12630`
* The shape of a traffic sign image:   `32x32`
* The number of unique classes/labels: `43`

#### 2. Distribution of the Data Set Across Labels

![Data Set Distribution][image1]

This data set does not have a equal distribution across all labels.

### Design and Test a Model Architecture

#### 1. Proprocessing Technique

First, the data are converted to grayscale image by averaging the three color channels.

**Original Images:**

![Original Img][image2]

**Result:**

![Grayscale][image3]

A more conventional method of normalization is to subtract each pixel by the mean and divide by the mean of the pixel value across the entire training set.

However I normalize the data set by using the following formula for each pixel:

`(pixel - 128) / 128`

I use this estimate because I want to avoid introducing more influence of the training set in my model when preprocessing the validation set and testing set (perhaps it is futile) to prevent overfitting.

The mean of the normalized data set is -0.354081335648, which is close to a zero-centered data, and the range of the pixel values are roughly between [-1, 1]


#### 2. CNN Architecture

The model is a LeNet with dropout layers added to the terminal fully connected layers.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale Image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6				|
| Convolution 5x5	    | 1x1 stride, outputs 10x10x16 									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16				|
| Fully connected		| 400 input nodes, outputs 120 nodes							|
| Drop Out              | Keep probability = 0.5
| Fully connected		| 120 input nodes, outputs 86 nodes							|
| Drop Out              | Keep probability = 0.5
| Fully connected		| 86 input nodes, outputs 43 nodes							|
| Softmax				| softmax across the 43 labels					|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I use the vanilla gradientDescentOptimizer for my model. The hyperparameters that I use for the solution are:

* Learning Rate: 0.001
* Batch Size: 128 (use powers of 2)
* Number of Epochs: 50 (I refrain using larger number of epochs because training time is extensively long)
* Mean: 0 (for initializing with random floats)
* Standard Deviation: 0.1 (for initializing with random floats)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

I kept a log of enlisting significant changes in my model while training it.

#### Log:
    1. 5/18/2018 - Training Accuracy 99.4%, Validation Accuracy 92.3%
        - preprocessing: simple normalization: (pixel - 128)/128
        - lr: 0.001, batch: 128, epoch: 50, mu: 0, sigma: 0.1
        - Seems to be overfitting
    2. 5/18/2018 - Training Accuracy 89.9%, Validation Accuracy 91.1%
        - added dropout at the fully connected layers
        - lr: 0.0015, batch: 128, epoch: 50, mu: 0, sigma: 0.1
    3. 5/18/2018 - Training Accuracy 94.2%, Validation Accuracy 92.6%
        - preprocessing: normalization and grayscaling
        - everything else remains the same
    4. 5/18/2018 - Training Accuracy 94.9% Validation Accuracy 94.1%
        - preprocessing: same
        - hyperparameter: everything stays the same except batch becomes 200
        - Try increasing learning rate 
    5. 5/19/2018 - Training Accuracy 85.0%, Validation Accuracy 86.7%
        - preprocessing: same
        - lr: 0.003 and everything else is the same
    6. 5/19/2018 - Training Accuracy 97.0%, Validation Accuracy 96.4%
        - preprocessing: find bugs in preprocessing: didn't divide properly
        - pixel - 128/128 instead of (pixel - 128)/128
        - lr: 0.001 and everything else is the same
        - Much better convergence


As a starting point, I use the LeNet as my model architecture. I refer to the LeNet lab for the initial hyperparameters. I achieve a high training accuracy (99%) but a lower validation accuracy (92%) in my first round. The overfitting issue is obvious, so I decide to add dropout layers to the fully connected layers in the convnet 

The result of adding dropout layers is immediate. The accuracy of my training and validation set come close to within 1%. However, the accuracy seems to plateau on around 92%. 

To increase the accuracy, I increase the learning rate to 0.0015 from 0.001. The result is insignificant, so I extend the number of epochs to 200 from 50 and have the experiment ran over night.

The accuracy improves to around 94%, so I decide to reduce the number of epoch back to 50 and double my learning rate to cut off the training time. However, the larger learning rate underfits the model at 85%. 

Instead, I turn to the preprocess stage to create better convergence. I realize that there is a bug in the preprocessing stage such that I did not properly normalize my data. Fittinng this issue immediate boost the model's performance to 96% validation accuarcy.

**Final model results were:**
* training set accuracy:   `97.6%`
* validation set accuracy: `96.9%`
* test set accuracy:       `94.2%`

Examining the test set accuracy reveals that the model is overfitting on both the training and validation set. A larger contributor to this issue, may be due to the fact that there is a unequal distribution of samples across each label in the data set. The model is perhaps underfitting these scarce labels. An improvement is to generate addition data for these scarce labels, by jittering images or tune up and down the brightness of the images.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Downloaded Images][image4] 

All of the images are in good quality with the objects of interest centered in the image. It should not be difficult for the model to make the right prediction. However, some of the traffic signs, including the childing crossing sign, has some complex shape, and at same time has scarce samples in the data set. It could pose a problem for the model to learn these shapes without a more complex convolution layers and larger number of samples.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	                | 
|:---------------------:|:---------------------------------:| 
| 30 km/h     		    | 30 km/h   						| 
| Child Crossing     	| Dangerous curve to the right	    |
| Keep Right		    | Keep Right					    |
| Road Work     		| Road Work			 				|
| Stop			        | Stop     							|


The model was able to correctly guess 4 of the 5 traffic signs. 80% accuracy is lower than the test accuracy, but this is under-sampling and we should see a higher accuracy if given more images. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

A closer look at the softmax probability yielded by the model's prediction on the label below:

    Image 1 - correct label: 1 - 'Speed limit (30km/h)'
       100.00%   Speed limit (30km/h)
         0.00%   Speed limit (20km/h)
         0.00%   Speed limit (70km/h)
         0.00%   Stop
         0.00%   End of all speed and passing limits

    Image 2 - correct label: 28 - 'Children crossing'
        83.90%   Dangerous curve to the right
        14.15%   Road work
         1.10%   Keep right
         0.34%   General caution
         0.29%   Children crossing

    Image 3 - correct label: 38 - 'Keep right'
       100.00%   Keep right
         0.00%   No passing
         0.00%   Speed limit (20km/h)
         0.00%   Speed limit (30km/h)
         0.00%   Speed limit (50km/h)

    Image 4 - correct label: 25 - 'Road work'
       100.00%   Road work
         0.00%   Dangerous curve to the right
         0.00%   Bicycles crossing
         0.00%   Slippery road
         0.00%   Right-of-way at the next intersection

    Image 5 - correct label: 14 - 'Stop'
        49.67%   Stop
        24.19%   Road work
         6.29%   Turn right ahead
         6.17%   Turn left ahead
         3.38%   Priority road


The model predicts with a great certainty on the '30km/h', 'Keep Right' and 'Road Work' signs. It's strange to see softmax output 100% on a label, but I believe that this could be just a round-off errors with the output decimal precision. 

The model is less certain on predicting the stop sign. The second candidate is road work. On its false prediction of the 'Children Crossing' sign as a 'Dangerous curve to the right' sign, we actually see that the correct label is one of the top 5 prediction at 0.29%, and the model is processing with noise.

The model seems to be able to detect and infer the prediction from simple shapes like numbers, arrows and letters. However, it has some difficulties distinguishing complex shape like a person, and perhaps overfits itself on detecting road work, because we can see the presence of the label as the second prediction candidate on lables with less certainty.

