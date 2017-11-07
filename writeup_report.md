** Behavioral Cloning Project **

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center_1.jpg "center image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2x2 filter sizes and depths between 12 and 54 (model.py lines 95-99) 

The model includes RELU layers to introduce nonlinearity (code lines 95-99), and the data is normalized in the model using a Keras lambda layer (code line 93). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 103 and 105). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 112-115). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 109).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to deepen the LeNet used in the previous traffic classification project.

My first step was to use a LeNet-like model with only three convolution layers and two fully connected layers with batch normalization and dropout. I also follow the advice of the cource material that I crop unrelated areas on the images which gave no benefit to the steering angle prediction.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a relatively high mean squared error on the training and validation set, so I decided to add more layers both on convolution and fully connected layers. With five convolution layers and four dense layers, I obtained the relatively low training error but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I checked the model which regularization methods gave low validation error. As a result, I found that the batch normalization doesn't work well on this task, and it gave lower validation error that I used the dropout with relatively high drop rate (0.7) after the first dense layer and low drop rate (0.3) after the second dense layer.

The final step was to run the simulator to see how well the car was driving around track one. There was only one spot where the vehicle fell off the track. I attribute this to my poor driving technique on the corner.
Largely, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 88-109) consisted of five convolution layers and four dense layers with dropout. This includes preprocessings of cropping the unrelated areas and pixel normalization (lines 90-93).


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded six laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to make use of full features from the three cameras on it. 

Then I repeated this process on each tracks in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would give another driving perspective data. 

After the collection process, I had 18822 number of data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by testing over 5-20 epochs, which results in 10 was enough. I used an adam optimizer so that manually training the learning rate wasn't necessary.