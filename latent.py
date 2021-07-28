import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plot
from matplotlib import style

path = 'recolab-data/latent/train.tit'
train_data = []
test_data = []
with open(path) as f:
    arr = f.read().split('\n')
    for line in arr[:-1]:
        word = line.split('\t')
        train_data.append([int(i) for i in word])

train_data = np.array(train_data)

path = 'recolab-data/latent/test.tit'
with open(path) as f:
    arr = f.read().split('\n')
    for line in arr[:-1]:
        word = line.split('\t')
	test_data.append([int(i) for i in word])

test_data = np.array(test_data)

class Latent_Factor():
    def __init__(self,R,K,A,B,iterations):
        self.R = R
        self.K = K
        self.A = A
        self.B = B
        self.num_items=R.shape 
        self.num_users=R.shape
        self.iterations = iterations
    
    def find_prediction(self, i, j):
        prediction = self.P1[i, :].dot(self.Q1[j, :].T)
        return prediction
        
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.find_prediction(i, j)
            error = 2*(r - prediction)
            self.P1[i, :] += self.A * (error * self.Q1[j, :] - 2*self.B * self.P1[i,:])
            self.Q1[j, :] += self.A * (error * self.P1[i, :] - 2*self.B * self.Q1[j,:])    

    def funct(self): 
        return self.P1.dot(self.Q1.T)

    def train(self):
        self.P1 = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q1 = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
	
        M = []
        num=10
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            error = self.error()
            M.append((i, error))
            if (i+1) % num == 0:
                print("Iteration: %d ; error = %.2f" % (i+1, error))

        return M
    
    def error(self):
         xs, ys = self.R.nonzero()
         predicted = self.funct()
         error = 0
         for i, j in zip(xs, ys):
             error += pow(self.R[i, j] - predicted[i, j], 2)
         return np.sqrt(error)

Z_matrix = np.max(train_data,axis = 0)
user = Z_matrix[1]
item = Z_matrix[0]
print(user,item)
choice = [10, 20, 40, 80, 100, 200]
R = np.zeros((user,item))

for i in train_data:
    R[i[1]-1,i[0]-1] = i[2]
print("Top 5")
print(R[:5]) 

for itr in choice:
    A = 0.01
    B = 0.005
    ans = Latent_Factor(R, K=itr, A=A, B=B, iterations=40)
    matrix=ans.train()
    error=0
    predicted=ans.funct()
    for l in test_data:
        error += (l[2] - predicted[l[1]-1, l[0]-1] ** 2)
    print('Test Error : ',error)
    matrix = np.array(matrix)
    style.use('ggplot')
    plot.figure(figsize=(12,6))
    plot.xlabel("Iteration")
    plot.ylabel("Mean Square Error")
    s = "Latent Factor : "+str(itr)+" lr = "+str(A) + ", lambda = " + str(B)
    plot.title(s)
    plot.plot(matrix[:,0],matrix[:,1])
    plot.show()
