#!/usr/bin/env python
# coding: utf-8
import numpy as np

##### GLOBAL VARIABLES ######
# eps is shift in numerical derivative [ (h(x+eps)-h(eps))/eps  ]
myeps=0.0001
# batchid keeps track with batch is being processed
# looped over in gradient function
batchid=0
# surpress the gradient of the weights
mylam=0.01



##### DATA CLASS ######

class data:
    """This is my data structure"""
    def __init__(self,myX,myY,bs=None,normalize=True):
        self.Xi = np.array(myX)
        self.Yi = np.array(myY)
        self.norm=normalize

        if bs == None or bs > len(self.Xi) :
            # if there is no batch size set it to full length of X
            self.bs=len(self.Xi)
        else:
            self.bs=bs
        # calculate number of batches
        self.batch_len=int(np.ceil(len(self.Xi)/self.bs))

        if  self.norm:
            # Define Mean and Standard Deviation
            self.X_max=np.max(self.Xi,axis=0)
            self.X_min=np.min(self.Xi,axis=0)
            self.Y_max=np.max(self.Yi,axis=0)
            self.Y_min=np.min(self.Yi,axis=0)
            #
            self.X_max[self.X_max == self.X_min] += 1
            self.Y_max[self.Y_max == self.Y_min] += 1
            # standardize and batch up
            self.Xno = norm(self.Xi, self.X_max, self.X_min)
            self.X = np.array_split(self.Xno, self.batch_len)
            self.Yno = norm(self.Yi, self.Y_max, self.Y_min)
            self.Y = np.array_split(self.Yno, self.batch_len)
        else:
            self.X = np.array_split(self.Xi, self.batch_len)
            self.Y = np.array_split(self.Yi, self.batch_len)
        #shuffle batches
        self.shuffle()
        # Find dimension of input layer
        self.dim=len(self.Xi[0])


    def shuffle(self):
        p=np.random.permutation(len(self.Xi))
        if self.norm:
            self.X = np.array_split(self.Xno[p], self.batch_len)
            self.Y = np.array_split(self.Yno[p], self.batch_len)
        else:
            self.X = np.array_split(self.Xi[p], self.batch_len)
            self.Y = np.array_split(self.Yi[p], self.batch_len)

    # def split(self,data):
    #     self.X=[]
    #     self.Y=[]
    #     for ele in data:
    #         xi,yi=np.split(ele,[-1],axis=1)
    #         self.X+=[xi]
    #         self.Y+=[yi]


##### MODEL CLASS ######

class model:
    """This will be my model"""
    def __init__(self,mydata):
        self.dat=mydata
        # We initialize with the input layer
        # dimension of input layer is determined by size of X
        self.act_dim=[len(self.dat.X[batchid][0])]
        # first layer is simply the input
        self.act_val=[self.dat.X[batchid]]
        # no weight matrix for input layer
        self.weight=[]
        # no gradient as well
        self.weight_grad=[]
        #no bias in the input layer
        self.bias=[]
        # no gradient
        self.bias_grad=[]
        # 0 for dense, 1 for conv
        # input not considered full layer
        self.layer_type=[]
        # no dropout in input layer
        self.drop=[]
        self.drop_rate=[]
        # list of types of activation functions (-1=identity,0=relu, 1=softmax, 2=sigmoid)
        # no activation function on input
        self.act_fct=[]


    def add_Dense(self,n,dropout=0.0,activation='relu'):
        # dim determined by previous activations dim
        input_dim=self.act_dim[-1]
        # add weight matrix and a placeholder for the grad
        self.weight+=[np.random.rand(input_dim,n)/input_dim]
        self.weight_grad+=[np.zeros((input_dim,n))]
        # add biases
        # bias size is automatically adjusted to batch size by numpy
        self.bias+=[np.zeros((1,n))]
        self.bias_grad+=[np.zeros((1,n))]
        # add dropout
        # dropout size is automatically adjusted to batchsize by numpy
        self.drop+=[np.random.binomial(1,1-dropout,size=(1,n))/(1-dropout)]
        self.drop_rate+=[dropout]
        self.act_dim+=[n]
        # 0 for dense layer (Important for forward propagation)
        self.layer_type+=[0]
        # add label for activation function
        if activation=='softmax':
            self.act_fct+=[1]
        elif activation=='sigmoid':
            self.act_fct+=[2]
        elif activation=='relu':
            self.act_fct += [0]
        else:
            print("Did not understand activation function. Using Relu")
            self.act_fct += [0]


    def forward_dense(self,ii):
        # Matrix multiply and add bias
        out=np.dot(self.act_val[ii],self.weight[ii])+self.bias[ii]
        # dropout
        out_drop=np.multiply(out,self.drop[ii])
        # apply non-linear function
        if self.act_fct[ii]==1:
            activation=softmax(out_drop)
        elif self.act_fct[ii]==2:
            activation=sigmoid(out_drop)
        else:
            activation=relu(out_drop)
        # save activations to use by backward propagation
        self.act_val+=[activation]

    def backward_dense(self,deltam_prev,ii):
        # build derivative of activation fct (deltam already includes weight matrix)
        if self.act_fct[ii]==1:
            # build matrix of derivatives from softmax deriv formula dS_i/dz_j=S_i(delta_ij-S_j)
            delta=[]
            for jj in range(len(self.act_val[ii+1])):
                Si=self.act_val[ii+1][jj]
                dadz=np.identity(len(Si))*Si-np.outer(Si,Si)
                delta+=[np.dot(deltam_prev[jj],dadz)]
            delta=np.array(delta)
        elif self.act_fct[ii]==2:
            sigx=sigmoid(self.act_val[ii+1])
            delta=np.multiply(deltam_prev,sigx*(1-sigx))
        else:
            # derivative of relu is heaviside function (matix has only diagonal entries can be multiplied directly)
            delta =np.multiply(deltam_prev,np.heaviside(self.act_val[ii+1], 0))
        # !! the iith element of activation function is actually the previous activation !!
        self.weight_grad[ii]=np.dot(self.act_val[ii].T,delta)
        self.bias_grad[ii]=np.sum(delta,axis=0,keepdims=True)
        # put already the weight matrix on it
        deltam=np.dot(delta,self.weight[ii].T)
        return deltam




    def forward_conv(self,ii):
        print('I dont know that yet')
        return None

    def backward_conv(self,delta_prev,ii):
        print('I dont know this yet')
        return None

    def forward(self,X):
        self.act_val=[X]
        for ii in range(len(self.layer_type)):
            if self.layer_type[ii]==0:
                self.forward_dense(ii)
            elif self.layer_type[ii]==1:
                self.forward_conv(ii)

    def backward(self,deriv_loss):
        # deltam is the regular delta already doted with the weight matrix
        deltam_prev=deriv_loss
        # go through all layers backwards
        for ii in range(len(self.layer_type)-1,-1,-1):
            # call different function depending on layer types
            if self.layer_type[ii]==0:
                # for dense
                deltam_prev=self.backward_dense(deltam_prev,ii)
            elif self.layer_type[ii]==1:
                # for conv
                deltam_prev=self.backward_conv(deltam_prev,ii)



    def predict(self,data=None):
        # Return model prediction,
        if data==None:
            # branch for data that model was initialized with
            # shuffle the dropped activations
            self.shuffle_drop(self.drop_rate)
            # forward propagate
            self.forward(self.dat.X[batchid])
            # return last activation
            return self.act_val[-1]
        else:
            # branch for new data (usually after model was trained on given data)
            mydat=np.array(data)
            # deactivate dropout
            old_drop=self.drop_rate
            self.shuffle_drop([0]*len(self.drop_rate))
            if self.dat.norm:
                # branch for normalize data
                myX=norm(mydat,self.dat.X_max,self.dat.X_min)
                # forward data
                self.forward(myX)
                # restore dropout
                self.shuffle_drop(old_drop)
                # denormalize last activation and return
                return norm(self.act_val[-1],self.dat.Y_max,self.dat.Y_min,forward=False)
            else:
                #non normalized branch
                # forward data
                self.forward(mydat)
                # restore dropout
                self.shuffle_drop(old_drop)
                # return last activation
                return self.act_val[-1]

    def shuffle_drop(self,drop_vec):
        self.drop=[np.random.binomial(1,1-drop_vec[ii],size=(1,self.act_dim[ii+1]))/(1-drop_vec[ii]) for ii in range(len(self.drop))]



##### LEARNER CLASS ######

class learner:
    """This will be my learner"""

    def __init__(self, mydata,mymodel,wd=0.01,al_mom=0.95,al_RMS=0.95,global_dropout=None,loss_function='mse'):
        self.mod=mymodel
        self.dat=mydata
        self.wd=wd
        self.amom=al_mom
        self.aRMS=al_RMS
        self.LossRec=np.array([])
        # Initial value for previous gradient and gradient square
        self.Mom_weight_p=[0*ele for ele in self.mod.weight]
        self.Mom_bias_p=[0*ele for ele in self.mod.bias]
        self.RMS_weight_p=[0*ele for ele in self.mod.weight]
        self.RMS_bias_p=[0*ele for ele in self.mod.bias]
        if global_dropout!=None:
            self.mod.drop_rate=[global_dropout for ii in self.mod.drop_rate]
        # Last layer should never have dropout
        self.mod.drop_rate[-1]=0
        if loss_function=='mse':
            self.loss_fct=1
        elif loss_function=='ce':
            self.loss_fct=2
        else:
            print("Lossfunction not recognized: Choose 'mse' or 'ce'")



    def learn(self,lr,epochs):
        for ii in range(epochs):
            for jj in range(self.dat.batch_len):
                batchid=jj
                self.optim(lr)
                # Record losses in LossRec
                self.LossRec=np.append(self.LossRec,self.loss())
                jj+=1
            # reshuffle batches
            self.dat.shuffle()
            ii+=1

    def cycle_learn(self,lr,epochs):
        if self.dat.batch_len<5:
            print("Too few batches using normal learn")
            self.learn(lr,epochs)
        else:
            # create vector of learning rates for cycle
            lr_vec=[dynamic_lr(lr,ii/self.dat.batch_len) for ii in range(self.dat.batch_len)]
            # create vector of momenta for cycle
            mom_vec=[dynamic_mom(self.amom,ii/self.dat.batch_len) for ii in range(self.dat.batch_len)]
            # create vector of RMS momenta for cycle
            RMS_vec = [dynamic_mom(self.aRMS, ii / self.dat.batch_len) for ii in range(self.dat.batch_len)]
            for ii in range(epochs):
                for jj in range(self.dat.batch_len):
                    batchid=jj
                    # set momentas according to place in cylce
                    self.amom=mom_vec[jj]
                    self.aRMS=RMS_vec[jj]
                    # call with varying learning rate
                    self.optim(lr_vec[jj])
                    # Record losses in LossRec
                    self.LossRec=np.append(self.LossRec,self.loss())
                    jj+=1
                # reshuffle batches
                self.dat.shuffle()
                ii+=1

    def loss(self):
        # Loss function
        if self.loss_fct==1:
            return mse(self.mod.predict(),self.dat.Y[batchid])
        elif self.loss_fct==2:
            return CrossEntropy(self.mod.predict(),self.dat.Y[batchid])

    def optim(self,lr):
        # predict returns model prediction for current batch
        yhat=self.mod.predict()
        if self.loss_fct==1:
            deriv_loss = 2 * (yhat - self.dat.Y[batchid])
        elif self.loss_fct==2:
            deriv_loss=-self.dat.Y[batchid]/yhat
        # backwards saves gradients into self.mod.weight_grad and self.mod.bias_grad
        self.mod.backward(deriv_loss)
        # calculate Momentum
        Mom_weight=self.Calc_Mom(self.mod.weight_grad,self.Mom_weight_p,self.amom)
        Mom_bias=self.Calc_Mom(self.mod.bias_grad,self.Mom_bias_p,self.amom)
        # calculate RMS
        RMS_weight=self.Calc_RMS(self.mod.weight_grad,self.RMS_weight_p,self.aRMS)
        RMS_bias = self.Calc_RMS(self.mod.bias_grad, self.RMS_bias_p, self.aRMS)
        # Update with RMS and Mom for Adam and weight decay
        for i in range(len(Mom_weight)):
            self.mod.weight[i]-=lr*(Mom_weight[i]/(RMS_weight[i]) +2*self.wd*np.sum(self.mod.weight[i]) )
        for i in range(len(RMS_weight)):
            self.mod.bias[i] -= lr * (Mom_bias[i] / (RMS_bias[i]))
        # save gradient and gradient square
        self.Mom_weight_p=Mom_weight
        self.Mom_bias_p = Mom_bias
        self.RMS_weight_p=RMS_weight
        self.RMS_bias_p = RMS_bias

    def Calc_Mom(self,grad,prev_grad,alpha):
        return [(1-alpha)*grad[i]+(alpha)*prev_grad[i] for i in range(len(grad))]

    def Calc_RMS(self,grad,prev_grad,alpha):
        if alpha==0:
            # set RMS to 1 if turned off
            RMS=1
        else:
            RMS=[((1-alpha)*grad[i]**2+(alpha)*prev_grad[i])**(1/2) for i in range(len(grad))]
            # Set all 0's to 1  in order to avoid div by 0
            # Essentially turns off RMS for this parameter
            for ele in RMS:
                ele[ele==0]=1
        return RMS

#### OLD VERSION OF GRADIENT WITH ACTUAL DERIVATIVE VERY VERY SLOW ####
    # def grad(self):
    #     #makes a list of gradients of loss function in respect to all parameters
    #     grad_array=[0*ele for ele in self.mod.weight]
    #     for ii in range(len(self.mod.weight)):
    #         for jj in range(len(self.mod.weight[ii])):
    #             for kk in range(len(self.mod.weight[ii][jj])):
    #                 self.mod.weight[ii][jj][kk]+=myeps
    #                 pred_eps=self.loss()
    #                 self.mod.weight[ii][jj][kk]-=myeps
    #                 pred=self.loss()
    #                 grad_array[ii][jj][kk]=(pred_eps - pred) / myeps
    #     return grad_array


##### EXTRA FUNCTIONS  ######

# Mean Squared Error
def mse(pred,val):
    diff=(pred-val)**2
    return (np.sum(diff))
#    return (np.sum(diff))/np.prod(diff.shape)

# Cross Entropy Loss
def CrossEntropy(pred,val):
    return -np.sum(val*np.log(pred))


# Standardize Data and Reversal of Standardization
def stand(array,mean,std,forward=True):
    # forward =True to standardize data
    # forward = False to transform standardized data back to its input form
    if forward:
        return (array-mean)/std
    else:
        return array*std+mean

def norm(array,max,min,forward=True):
    # forward =True to normalize data
    # forward = False to transform normalize data back to its input form
    if forward:
        return (array-min)/(max-min)
    else:
        return min+array*(max-min)


# Compute Dynamic lr for cycle learning
def dynamic_lr(lr,x,lr_min=None):
    if lr_min==None:
        lr_min=lr*0.05
    if x<=0.5:
        # learning rate grows linearly from lr_min to lr
        return 2*(lr-lr_min)*x+lr_min
    elif x<=1:
        # learning rate decays like sin from lr to lr_min
        return (lr-lr_min)/2*np.sin(np.pi*(2*x-0.5))+(lr+lr_min)/2
    else:
        return print("Error in dynamic_lr: given x out of range")

# Compute Dynamic momenta for cycle learning
def dynamic_mom(mom,x,mom_min=None):
    if mom_min==None:
        mom_min=mom*0.9
    if x<=0.5:
        # mom decays linearly from mom to mom_min
        return 2*(mom_min-mom)*x+mom
    elif x<=1:
        # mom decays like sin to mom_min
        return (mom_min-mom)/2*np.sin(np.pi*(2*x-0.5))+(mom+mom_min)/2
    else:
        return print("Error in dynamic_lr: given x out of range")

#relu function for non-linearity
def relu(x):
    return np.maximum(0,x)

# softmax for non-lonearity
def softmax(x):
    return np.exp(x)/(np.sum(np.exp(x),axis=1,keepdims=True))

def sigmoid(x):
    return 1/(1+np.exp(-x))


