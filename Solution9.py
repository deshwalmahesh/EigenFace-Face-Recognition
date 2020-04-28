
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from math import sqrt
from math import pi
from math import exp
import pandas as pd
from sklearn.model_selection import train_test_split



class BAYES():
    '''
    class to implement the bayes theorem
    '''
    def __init__(self,dataset):
        '''
        dataset should be 2-D array of with last column being the LabelEncoded target class
        [[111,222,0],
         [222,111,1]]
        '''
    
        self.dataset = dataset
    
    
    def separate_by_class(self):
        '''
        returns a dictonary of of {class_label:[data_points]}
        '''
        dataset=self.dataset
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated
 

    def mean(self,numbers):
        '''
        Calculate the mean of a data point (list of numbers)
        '''
        return sum(numbers)/float(len(numbers))
 
   
    def stdev(self,numbers):
        '''
        Calculate the standard deviation of data point (list of numbers)
        '''
        avg = self.mean(numbers)
        variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
        return sqrt(variance)

   
    def summarize_dataset(self,dataset):
        '''
        Calculate the mean, stdev and count for each column in a dataset
        '''
        # dataset=self.dataset
        summaries = [(self.mean(column), self.stdev(column), len(column)) for column in zip(*dataset)]
        del(summaries[-1])
        return summaries

    
    def summarize_by_class(self):
        '''
        Split dataset by class to get statistics for each row or data point
        '''
        dataset=self.dataset
        separated = self.separate_by_class()
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    
    def calculate_probability(self,x, mean, stdev):
        '''
        Calculate the Gaussian PDF for any point 
        '''
        exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent
    

    def calculate_class_probabilities(self,summaries, row):
        '''
        Calculate the probabilities of belonging to each class for a given row or data point
        '''
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(row[i], mean, stdev)
        return probabilities
    
 
    def fit_train(self):
        '''
        calculate all the necessary implementations to fit the data first
        '''
        dataset = self.dataset
        self.summaries = self.summarize_by_class()
        
        
    def predict_prob(self,test_point):
        '''
        predict probabilities for each given in dataset for the data point
        '''
        return self.calculate_class_probabilities(self.summaries, test_point)
    
    
    
    def predict_label(self,test_point):
        '''
        predict label for the data point
        '''
        prob = self.calculate_class_probabilities(self.summaries, test_point)
        maxi = -999
        key = False
        curr_label = False
        for key in prob:
            if prob[key]>maxi:
                maxi = prob[key]
                curr_label = key
        return(curr_label)


# In[249]:


df = pd.read_csv('usps-2cls.csv',header=None)
df.head()


# In[241]:


X = df.iloc[:,:256].values
y = df.iloc[:,256].values.reshape(-1,1)


# In[243]:


for ratio in [0.1,0.2,0.5,0.8,0.9]:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=ratio)
    dataset = np.append(X_train,y_train,axis=1)
    


# In[244]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2) 
dataset = np.append(X_train,y_train,axis=1)


# In[245]:


def predict(train_data,test_data):
    bayes = BAYES(train_data)
    bayes.fit_train()
    y_pred = []
    for i in range(len(test_data)):
        y_pred.append(bayes.predict_label(test_data[i]))
    return y_pred

