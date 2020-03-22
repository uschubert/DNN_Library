# DNN_Library

## Overview
Python library for deep neural networks based on numpy.    

## Usage
The package has a data class, a model class and together they will be loaded into a learner class, which can be used to train the model. A more detailed description of each class can be found below. 

### Data Class
The data should be loaded into the data class with the command: \
data(X,Y,bs=500,normalize=True) \
The features $X and the predictions $Y will be converted into numpy arrays and normalized (if $normalize is set to true). Finally the data is shuffeld and divided into batches according to the batch size ($bs). After each epoch the data is shuffeld and divided into batches to ensure a well randomized data set.

### Model Class
The model can be initialized by loading the data into the model class with the command: \
model(data) \
Next we can add a fully connected layers with the command: \
model.add_Dense(hidden_size,dropout=0.0,activation='relu') \
The parameter $hidden_size determines the size of the internal layer and the activation functions determines the non-linearity that is being applied. Currently there are three non-linear functions implemented: 'relu', 'softmax' and 'sigmoid'. One can set a specific dropout rate with $dropout for this layer, which will randomly deactivate certain nodes during training. Dropout will be deactivated in the test phase. Please note that the last layer needs to have the same dimension as the dimension of the prediction $Y. \
The predict routine can be used to make predictions of the model on data specified by $test_data \
model.predict(test_data)\



### Learner Class
The learner can be created with the following command: \
learner(data,model,wd=0.0,al_mom=0.95,al_RMS=0.95,global_dropout=0,loss_function='mse') \
The learner controls all routines that are needed for the training of the model and the loss function that is used. For the loss function one can use 'mse' for the mean squared error and 'ce' for cross entropy loss. The parameter $wd sets the rate of weight decay that is used during the training. The learning rate can be dynamically adjusted through $al_mom, which sets the momentum and $al_RMS, which sets the RMSprop. In addtion $global_dropout can be used to set a global dropout rate for all layers. This option will override any dropout rates that were set during the model building. \
Once the learner is created we can train the model through\
learner.learn(learning_rate,epochs) \
The $learning_rate sets the learning rate that is used and $epochs refers to the number of times the model goes through all data batches for the training.\
Additionally one can use cycle learn through the command: \
learn.cylce_learn(learning_rate,epochs)\ 
Cycle learn will linearly rise the learning rate from $learning_rate/10 to the full $learning_rate and then back to $learning_rate/10, while $al_mom and $al_RMS go through the opposite cycle first dropping linearly and then rising back to the specified amount.  


## Implemented Concepts
- Dense layers
- Activation functions (ReLu, Sigmoid, SoftMax)
- Loss functions (Mean squared error, Cross entropy loss)
- Back propagation
- Adam optimizer
- One cycle fit
- Dropout
- Weight decay

## Possible Extensions
- Convolutional layers
- Batch normalization
- Not fully connected layers
- Layer-wise relevance propagation
- and many more


