import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import pickle
import time

np.random.seed(62501)

class DeepNeuralNetwork:
    L = 3
    W = None
    b = None
    n_output_neurons = 2
    
    def __init__(self, n_input_features, n_output_neurons, n_hidden_layers_neurons):
        self.n_output_neurons = n_output_neurons
        n_hidden_layers_neurons.append(n_output_neurons)
        self.L = len(n_hidden_layers_neurons)
        self.W = list()
        self.b = list()
        n = [n_input_features]+n_hidden_layers_neurons
        for i in range(0,self.L):
            low = -2.0/np.sqrt(n[i+1])
            high = 2.0/np.sqrt(n[i+1])
            self.W.append(np.random.uniform(low,high,(n[i],n[i+1])))
            self.b.append(np.random.uniform(low,high,(1,n[i+1])))
            
    def set_params(self,w_file_name,b_file_name):
        w_file = open(w_file_name,'rb')
        b_file = open(b_file_name,'rb')
        self.W = pickle.load(w_file)
        self.b = pickle.load(b_file)
        w_file.close()
        b_file.close()
        
    
    def g(self,z, act_func):
        z = np.clip(z,-10,10)
        if act_func == 'sigmoid':
            return 1.0/(1.0+np.exp(-z))
        elif act_func == 'tanh':
            z = 2*z
            s = 1.0/(1.0+np.exp(-z))
            return 2*s - 1.0
        elif act_func == 'relu':
            return z * (z > 0)

    
    def g_prime(self,z,act_func):
        a = self.g(z)
        if act_func == 'sigmoid':
            return np.multiply(a,(1.0-a))
        elif act_func == 'tanh':
            return 1-np.multiply(a,a)
        else:
            return 1.0 * (z > 0)
        
    
    def softmax(self,a):
        z = np.exp(a - a.max(axis=1).reshape((a.shape[0],1)))
        deno = z.sum(axis=1).reshape((z.shape[0],1))
        s = np.divide(z,deno)
        return s
    
    def compute_cost(self,X,Y):
        h = X
        for i in range(0,self.L-1):
            a = np.dot(h,self.W[i]) + self.b[i]
            h = self.g(a)
        a = np.dot(h, self.W[self.L-1]) + self.b[self.L-1]
        O = np.log(self.softmax(a))
        m = len(X)
        return (-1.0/float(m))*np.sum(np.multiply(Y,O))
        
    def propagate(self,X,Y,act_func):
        h = [None]*(self.L)
        
        #initialize h0 to X
        h[0] = X
        
        #forward propagation
        for i in range(0,self.L-1):
            a = np.dot(h[i], self.W[i]) + self.b[i]
            h[i+1] = self.g(a, act_func)
        
        a = np.dot(h[self.L-1], self.W[self.L-1]) + self.b[self.L-1]
        O = self.softmax(a)
        
    
        #back propagation
        dW = list()
        db = list()
        dLda = -1*(Y - O) 
        for i in range(self.L-1, -1,-1):
            
            dLdW = np.dot(h[i].T,dLda)
            dLdb = np.sum(dLda,axis=0)
            dLdb = dLdb.reshape((1,dLdb.shape[0]))
            dW = [dLdW]+dW
            db = [dLdb]+db
            
            dLdh = np.dot(dLda,self.W[i].T)
            dLda = np.multiply(dLdh,np.multiply(h[i],(1.0-h[i])))
            
        return (dW,db)
      
      
    def fit(self,X,Y,lr,epochs, act_func):
        costs = list()
        for e in range(0,epochs):
            dW,db = self.propagate(X,Y)
            for i in range(0,self.L):
                self.W[i] = self.W[i] - (lr*dW[i])
                self.b[i] = self.b[i] - (lr*db[i])
            c = self.compute_cost(X,Y)
            costs.append(c)
            print 'Cost after '+str(e+1)+' epochs : '+str(c)
        return costs
                
    def predict(self, X):
        h = X
        for i in range(0,self.L-1):
            a = np.dot(h,self.W[i]) + self.b[i]
            h = self.g(a)
        a = np.dot(h, self.W[self.L-1]) + self.b[self.L-1]
        O = self.softmax(a)
        return np.argmax(O,axis=1)
      
    def compute_accuracy(self,y_predict,y_actual):
        hits = 0
        for i in range(0,len(y_predict)):
            if y_predict[i] == y_actual[i]:
                hits += 1
        return float(hits)/float(len(y_actual))
              
        
data = pd.read_csv('apparel_train.csv')
data.head()
input_cols = list()
for i in range(1,785):
    input_cols.append('pixel'+str(i))
data_train, data_test, label_train, label_test = train_test_split(
    data[input_cols],
    data[['label']],
    test_size=0.2,
    random_state=0)
X_train = data_train.values
X_test = data_test.values
y_tr_temp = label_train.values
y_ts_temp = label_test.values
y_tr_temp = y_tr_temp.reshape((y_tr_temp.shape[1],y_tr_temp.shape[0]))
index = y_tr_temp[0]
y_train = np.zeros((y_tr_temp.shape[1],10))
y_train[np.arange(y_tr_temp.shape[1]),index] = 1

y_ts_temp = y_ts_temp.reshape((y_ts_temp.shape[1],y_ts_temp.shape[0]))
index = y_ts_temp[0]
y_test = np.zeros((y_ts_temp.shape[1],10))
y_test[np.arange(y_ts_temp.shape[1]),index] = 1

dnn = DeepNeuralNetwork(784,10,[10,10,10])
dnn.fit(X_train,y_train,1e-5,1000,'sigmoid') 
y_test_pred = dnn.predict(X_test)
print dnn.compute_accuracy(y_test_pred,label_train['label'].tolist())    
        
        
    
