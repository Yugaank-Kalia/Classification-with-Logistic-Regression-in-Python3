import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# loading data 
data = np.loadtxt('machine-learning-ex2/ex2/ex2data1.txt', delimiter = ',')

# creating training sets 
x = np.array(data[:,0:2])
y = np.c_[data[:,2]]

# adding layer of bias

def normalize(x):
    
    return (x - np.mean(x))/(np.amax(x)-np.amin(x))

X = np.c_[np.ones((x.shape[0],1)),normalize(x)]

# m = length of training set
# n = number of features 
m = X.shape[0]
n = X.shape[1]

# setting number of iterations and learning rate, α and Initialises the weights to ZERO  
iter = 15000
alpha = 0.03
theta = np.zeros((n,y.shape[1]))

# plots the dataset using matplotlib.pyplot
def visualize_data(x, y, z):
    
    plt.scatter(x, y, marker = 'o', c=z)
    plt.show()

visualize_data(x[:, 0], x[:, 1], np.squeeze(y, axis = -1))

def sigmoid(z):
    return 1./(1+np.exp(-z))

def cost(theta,X,y):
    
    hyp = sigmoid(np.dot(X,theta))
    
    positive = np.multiply(y, np.log(hyp))
    negative = np.multiply((1-y), np.log(1-hyp))

    return (-1./m)*np.sum(positive+negative)

def gradDescent(theta,X,y):
    
    costs = list()

    for i in range(iter):

        hyp = sigmoid(X.dot(theta))

        gradient = (X.T.dot(hyp-y))
        theta = theta - (alpha/m)*gradient

        costs.append(cost(theta,X,y))

    return theta,costs

theta,costs = gradDescent(theta,X,y)

line = X.dot(theta)

def visualize_cost():
    
    y_axis = np.array(costs)
    x_axis = np.array([i for i in range(iter)])
    plt.scatter(x_axis,y_axis)
    plt.show()

visualize_cost()

def decisonBoundary(prob):
    temp = list()
    for i in range(len(prob)):
        if(prob[i] >= 0.5):
            temp.append(np.array(1))
        else:
            temp.append(np.array(0))
    return np.array(temp)


def visualize_plot(x,y,z):
    np.random.seed(19680801)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=np.squeeze(z,-1))
    ax.scatter(x, y, line, marker='_')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y (Labels)')

    plt.show()

visualize_plot(x[:, 0], x[:, 1], y)
