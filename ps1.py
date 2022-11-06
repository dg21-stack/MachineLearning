## Daniel Gaidar CSCI3349, ps1
## Description : load the digits data set and perform N stochastic gradient descents (in this case 50) over M random data examples (in this case 250) selected each iteration.

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import numpy as np
import random

def trainDigit(rng = 50,samples = 250):
    digits = load_digits()
    a = []
    X = []
    y = []
    xAxis = range(1,rng + 1)
    finData = []
    clf = SGDClassifier()
    for i in range(0, rng):
        a = random.choices(range(0,len(digits.data)), k =  samples)
        for j in range(0, samples): 
            X.append(digits.data[a[j]])
            y.append(digits.target[a[j]])
        Xnp = np.array(X)
        ynp = np.transpose(np.array(y))
        clf.partial_fit(Xnp,ynp, classes = np.unique(ynp))
        finData.append(clf.score(Xnp,ynp))
        a = []
        X = []
        y = []
    
    plt.plot(xAxis, finData, "-d", label = "Accuracy for Training Iteration")
    plt.ylabel('Classification Accuracy')
    plt.xlabel('Training Iterations')
    plt.title('Classification Accuracy over 50 Training Iterations')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    trainDigit()
