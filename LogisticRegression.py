import numpy as np
import matplotlib.pyplot as plt

# loading data 
data = np.loadtxt('machine-learning-ex2/ex2/ex2data1.txt', delimiter = ',')

# creating training sets 
x = data[:,0:2]
y = np.c_[data[:,2]]

# adding layer of bias
X = np.c_[np.ones((x.shape[0],1)),x]

# m = length of training set
# n = number of features 
m = X.shape[0]
n = X.shape[1]

# setting number of iterations and learning rate, α and Initialises the weights to ZERO  
iter = 1500
alpha = 0.03
theta = np.zeros((n,y.shape[1]))

# plots the dataset using matplotlib.pyplot
'''plt.scatter(x[:, 0], x[:, 1], marker='X', c=y)
plt.show()'''

def sigmoid(z):
    return 1./(1+np.exp(-z))

def cost(theta,X,y):
    for i in range(iter):
        hyp = sigmoid(np.dot(X,theta))
        positive = np.multiply(y,np.log(hyp))
        negative = np.multiply((1-y),np.log(1-hyp))

    return (-1./m)*np.sum(positive+negative)

def gradDescent(theta,X,y):
    for i in range(iter):
        hyp = sigmoid(X.dot(theta))
        gradient = (X.T.dot(hyp-y))
        theta = theta - (alpha/m)*gradient
    return theta

def decisonBoundary(prob):
    return 1 if prob>=0.5 else 0

def classifier(predictions):
    classify = list()
    for i in predictions:
        classify.append(decisonBoundary(i))
    return classify
