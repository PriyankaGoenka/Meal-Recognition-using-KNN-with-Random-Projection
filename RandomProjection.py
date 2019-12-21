#Random Projection to reduce the features
import numpy as np
import random
from numpy import genfromtxt
my_data = genfromtxt('Dataset.csv', delimiter=',')
#my_data[0, 0]=30
print(np.shape(my_data))
Dataset=my_data
Dataset=Dataset[0:,0:2500]
print(np.shape(Dataset))

def RandomProjection(size,features):
    I = np.zeros([2500, 1])
    for i in range(features):
        random = np.random.normal(0, 1, (size, 1))
        I = np.concatenate((I, random), axis=1)
    I = I[0:, 1:]
    print(np.shape(I))
    return I
#print(I)


reduced_features = np.matmul(Dataset, RandomProjection(np.shape(Dataset)[1], 100))
#print(np.shape(reduced_features))
np.savetxt("DatasetReduced.csv", reduced_features, delimiter=",", fmt='%s') # Save the matrix in csv format