'''
ONE VS ALL FOR MELANOMA DETECTION.
Luca Zammataro, 2020

an exercise for trying to automatically detect Melanoma 
original dermoscopic images from PH2: https://www.fc.up.pt/addi/ph2%20database.html

Use the follwing files: PH2_128X128.pickle, or PH2_128X128_BIVA.pickle (filtering out all the redundant features by Bivariate analysis)
More information here: https://towardsdatascience.com/detecting-melanoma-with-one-vs-all-f84bc6dd0479

Warranty disclaimer:
The only aim of this exercise is to expose basic concepts on the application of Machine Learning 
to images of skin lesions, and it is intended only for experimental purposes, not for clinical use.

'''

'''
LOAD ALL THE NECESSARY PYTHON LIBRARIES
'''
import cv2 # OpenCV-Python is a library of Python bindings designed to solve computer vision problems. https://docs.opencv.org/master/d0/de3/tutorial_py_intro.html
import numpy as np # a robust library of math functions for scientific computing with Python. https://numpy.org/
from matplotlib import pyplot as plt # Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. https://matplotlib.org/
import pandas as pd #pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language. https://pandas.pydata.org/
import pickle # pickle: Python object serialization. https://docs.python.org/3/library/pickle.html
import scipy.optimize as opt # scipy is a Python-based ecosystem of open-source software for mathematics, science, and engineering. https://www.scipy.org/
import scipy.io as sio
import random # random: Generate pseudo-random numbers. https://docs.python.org/3/library/random.html
from random import randint
from sklearn.metrics import matthews_corrcoef #Scikit-learn is an open source machine learning library that supports supervised and unsupervised learning. https://scikit-learn.org/stable/


'''
REGULARIZED LOGISTIC REGRESSION FUNCTIONS
'''

## LOGISTIC FUNCTION
def sigmoid(z): 
    return np.float64(1 / (1 + np.exp(-z)))


# REGULARIZED LOGISTIC REGRESSION COST FUNCTION 
def calcLrRegCostFunction(theta, X, y, lambd):

    
    # number of training examples
    m,n = X.shape  
    
    # Calculate h = X * theta (we are using vectorized version)
    h = X.dot(theta) 
    
    
    # Calculate the Cost J
    J = (np.sum(np.multiply(-y,np.log(sigmoid(h))) - \
                 np.multiply((1.0 - y),np.log(1.0 - sigmoid(h)))) /m) + \
                 np.sum(theta[1:]**2) * lambd / (2 * m)    
    
    

    return np.float64(J)



# REGULARIZED LOGISTIC REGRESSION GRADIENT
def calcLrRegGradient(theta, X, y, lambd):
    
    # number of training examples
    m,n = X.shape  
    
    # Calculate h = X * theta (we are using vectorized version)
    h = X.dot(theta) 
    
    # Calculate the error = (h - y)
    error = np.subtract(sigmoid(h), y)    
    
    # Calculate the new theta
    theta_temp = theta
    theta_temp[0] = 0.0
    gradient = np.sum((((X.T).dot(np.divide(error,m))), theta_temp.dot(np.divide(lambd,m)) ), axis=0   )
    

    return np.float64(gradient)



'''
ONE VS ALL FUNCTIONS
'''


def oneVsAll(X, y, lambd):    
    
    print('-ONE VS ALL-')
    
    num_labels = len(set(y))
    m , n = X.shape;
    all_theta = np.array(np.zeros(num_labels * (n+1))).reshape(num_labels,n+1)
    initial_theta = np.zeros(n+1)
    
    # Add a column of 'ones' to X
    # Add a column of ones to the X matrix
    X = np.vstack((np.ones(m), X.T)).T        


    for i in range(0, num_labels):
        in_args = (X, (( y == i).astype(int)), lambd)
        theta = opt.fmin_cg(calcLrRegCostFunction, initial_theta, \
                           fprime=calcLrRegGradient, args=in_args, gtol=1e-4, \
                            maxiter=500, full_output=False)
        all_theta[i:] = theta.T

    print('\n')    
    return all_theta



def predictOneVsAll(all_thetas, X, y):
    m , n = X.shape
    X = np.vstack((np.ones(m), X.T)).T    
    # This line calculate the max Theta
    prediction = np.argmax(sigmoid( np.dot(X,all_thetas.T) ), axis=1)
    print('Accuracy: {:f}'.format( ( np.mean(prediction == y )*100 ) ) )
    return prediction



'''
IMAGE VISUALIZATION FUNCTION
'''

def Display(df, n):

    res = int(np.sqrt(len(df['X'][0])))
    w, h = (res, res)
    image = df['X'].values[n].reshape(w,h)
    plt.imshow(image, cmap="gray"), plt.axis("off"), \
    plt.title(["#"+str(n), df['ID'].values[n], "outcome="+str(df['y'].values[n])])
    

    
'''
UPLOAD DATASET FUNCTION
'''

def UploadDataset(fileName):

    ArchiveName=fileName

    with open(ArchiveName, 'rb') as handle: df_PH2 = pickle.load(handle)

    Features = df_PH2['X'].values
    y = df_PH2['y'].values    

    featureList = []

    for i in range(len(Features)):
        featureList.append(Features[i].tolist())

    X = np.asarray(featureList)    

    print('-UPLOADING DATASET-')
    print('archive:', ArchiveName)
    print('num of examples:', len(X))
    print('num of features (X):', len(X[0]))
    print('y is the output vector')
    print('num of classes:', len(set(y)))
    print('\n')    
    

    return [df_PH2, X, y]


'''
SPLIT DATASET FUNCTION
'''

def splitDataset(X, y):
    # Split the Dataset into training and testing set.
    # Make a random permutation, to split the data randomly

    print('-SPLITTING DATASET-')
    rs = random.randint(0, 100000)
    np.random.seed(rs)


    num_test = round(len(X)/5)

    # Randomize indices based on the X length
    indices = np.random.permutation(len(X))

    # the training set is composed by all but the test examples
    X_train = X[indices[:-num_test]]
    y_train = y[indices[:-num_test]]

    # select the last 1/3 (reserved to  the test)
    X_test = X[indices[-num_test:]]
    y_test = y[indices[-num_test:]]

    print('random seed:',rs)
    print('# Training set: ', len(X_train))
    print('# Test set: ', len(X_test))
    print('\n')    
    

    # return Training and Testing set, a list with random indices, and number of tests
    return [X_train, y_train, X_test, y_test, indices, num_test]


'''
CALCULATE ACCURACY FUNCTION
'''

def CalcAccuracy(all_thetas, X_train, y_train, X_test, y_test, num_test ):
    # Calculate Training Accuracy
    print('-TRAINING ACCURACY-')
    pred_train= predictOneVsAll(all_thetas, X_train, y_train)
    print('matthews_corrcoef:', matthews_corrcoef(y_train.tolist(), pred_train.tolist()))
    print('\n')

    # Calculate Test Accuracy
    print('-TEST ACCURACY-')
    pred_test= predictOneVsAll(all_thetas, X_test, y_test)
    print('matthews_corrcoef:', matthews_corrcoef(y_test.tolist(), pred_test.tolist()))
    print('\n')    


    # Make a Dataframe with predictions
    test_bench = pd.DataFrame([indices[-num_test:], \
                  y_test.tolist(), \
                  pred_test.tolist()]).T
    test_bench.columns = ['ID', 'y', 'prediction']
    
    return test_bench, pred_train, pred_test



'''
SHOW THE MAX THETA FUNCTION
'''

def showTheMaxTheta(X, y, all_thetas, imageID, pred):

    # Make a copy of X
    X_original = X
    m , n = X.shape
    X = np.vstack((np.ones(m), X.T)).T    
    MaxTheta = max(sigmoid(np.dot(X[imageID],all_thetas.T)))
    
    # apply all the theta matrix on a specific X
    MaxThetaPosition = sigmoid(np.dot(X[imageID],all_thetas.T))

    # make a Dataframe
    MaxThetaDf = pd.DataFrame(MaxThetaPosition.tolist()) 
    for col in MaxThetaDf.columns:
        predictedCategory = MaxThetaDf[MaxThetaDf[col] == MaxTheta].index.tolist()
    
    print(str("Real outcome: "+str(y[imageID])))
    print(str("Max Theta: "+str(MaxTheta)))        
    print(str("Predicted label: "+str(predictedCategory[0])))
    print ("\n")
    print(MaxThetaDf)
    
    return 




'''
MAIN
'''

# Upload the dataset
df, X, y = UploadDataset('PH2_128X128_BIVA.pickle')

# Split the dataset
X_train, y_train, X_test, y_test, indices, num_test = splitDataset(X, y)

# Run oneVsAll
lmbda = np.float64(1e-4)
all_thetas = oneVsAll(X_train, y_train, lmbda)

# Calc Accuracy
df_test_bench, pred_train, pred_test = CalcAccuracy(all_thetas, X_train, y_train, X_test, y_test, num_test)

# Print the test results
df_test_bench
