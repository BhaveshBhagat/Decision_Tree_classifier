#!/usr/bin/env python
# coding: utf-8

# # Iris Data Prediction using Decision Tree Algorithm

# Task - 
# The Purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly..

# In[1]:


#Importing multiple library to read,analysed and visualized the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Reading the Iris.csv file 

Iris_data = pd.read_csv('Iris.csv')


# # Some Basic Information of Data set

# In[3]:


#Checking top 10 records of Dataset..
Iris_data.head(10)


# In[4]:


#Basic Information regarding data

Iris_data.info()


# Iris_data contain total 6 features in which 4 features(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalwidthCm) are independent features and 1 feature(Species) is dependent or target variable.
# 
# All Independent features has not-null float values and target variable has class labels(Iris-setosa, Iris-versicolor, Iris-virginica)

# In[5]:


#Describe function gives the basic numerical info about data for each numeric feature..

Iris_data.describe()


# In[6]:


#Data points count value for each class labels..

Iris_data.Species.value_counts()


# As we saw that each classes has equal number of data points then our Iris data said to be Balanced dataset. No Class is fully dominating in our dataset.

# # Visualizing Iris Data

# For Visualizing the dataset we used Matplotlib or seaborn as a python library.
# Their are many plots like scatter, hist, bar, count etc. to visualized the data for better understanding...  

# In[7]:


#Visualizing the dataset features to find pattern to solve our task

plt.scatter(Iris_data['SepalLengthCm'],Iris_data['SepalWidthCm'])
plt.show()


# In[8]:


#Using Seaborn lib to visualized 2 features based on target variable.

sns.set_style('whitegrid')
sns.FacetGrid(Iris_data, hue = 'Species')    .map(plt.scatter, 'SepalLengthCm','SepalWidthCm')    .add_legend()

plt.show()


# BY looking the Scatter plot we can say that all bluepoints(Iris-setosa) are separated perfectly as compare to orange(versicolor) or green(virginica) points for features(SepalLengthCm, SepalwidthCm) 

# In[9]:


#Pair plot gives the relationship b/w all features distribution with each other..

sns.pairplot(Iris_data.drop(['Id'],axis=1), hue='Species')
plt.show()


# BY looking the result of pair plot we sure that all blue points are well separated with other two classes. But Versicolor and virginica are partially overlapping with each other.
# 
# In pair plot we saw that their are some feature combination which has very less overlapping b/w Versicolor and verginica, that's means those feature are very important for our classification task purpose.

# # Exploring Some New Features
# 
# Here I just try to find some new feature with the help of exisiting features. 
# 
# -Taking difference of each feature with each other to get some more information and visualized it by using plots. 

# In[10]:


#Just trying to explore some new feature using the given data...

Iris_data['Sepal_diff'] = Iris_data['SepalLengthCm']-Iris_data['SepalWidthCm']
Iris_data['petal_diff'] = Iris_data['PetalLengthCm']-Iris_data['PetalWidthCm']
Iris_data


# In[11]:


#Analysed new feature to get some more infomation apart form existing ones...

sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue='Species')   .map(plt.scatter,'Sepal_diff','petal_diff')   .add_legend()
plt.show()    


sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue='Species')   .map(sns.distplot,'petal_diff')   .add_legend()
plt.show()    


# In[12]:


Iris_data['Sepal_petal_len_diff'] = Iris_data['SepalLengthCm']-Iris_data['PetalLengthCm']
Iris_data['Sepal_petal_width_diff'] = Iris_data['SepalWidthCm']-Iris_data['PetalWidthCm']
Iris_data


# In[13]:


sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue='Species')   .map(plt.scatter,'Sepal_petal_len_diff','Sepal_petal_width_diff')   .add_legend()
plt.show()

sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue='Species')   .map(sns.distplot,'PetalLengthCm')   .add_legend()
plt.show()


# In[14]:


Iris_data['Sepal_petal_len_wid_diff'] = Iris_data['SepalLengthCm']-Iris_data['PetalWidthCm']
Iris_data['Sepal_petal_wid_len_diff'] = Iris_data['SepalWidthCm']-Iris_data['PetalLengthCm']
Iris_data


# In[15]:


sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue='Species')   .map(plt.scatter,'Sepal_petal_wid_len_diff','Sepal_petal_len_wid_diff')   .add_legend()
plt.show()

sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue='Species')   .map(sns.distplot,'Sepal_petal_wid_len_diff')   .add_legend()
plt.show()


# In[16]:


# Finding relationship b/w new feature based on class labels...

sns.pairplot(Iris_data[['Species', 'Sepal_diff', 'petal_diff', 'Sepal_petal_len_diff',       'Sepal_petal_width_diff', 'Sepal_petal_len_wid_diff',       'Sepal_petal_wid_len_diff']], hue='Species')
plt.show()


# - With help of Pair plot we are getting some new information but it is more likely similar with our main data features as we saw earlier.
# 
# Every combination well separate the Iris-setosa but has some overlapped b/w Versicolor and virginica.

# In[17]:


#Droping Id column as it is of no use in classifing the class labels..

Iris_data.drop(['Id'],axis=1,inplace=True)


# Checking distribution plot for each feature in dataset for each class label... 

# In[18]:


#exploring distribution plot for all features

for i in Iris_data.columns:
    if i == 'Species':
        continue
    sns.set_style('whitegrid')
    sns.FacetGrid(Iris_data,hue='Species')    .map(sns.distplot,i)    .add_legend()
    plt.show()


# # Building Classification Model

# In[19]:


#Now try to create a model to solve our task
#As per our analysis, we can't find much information from new feature which can helpful in solving our problem...
#For solving our task I have selected few features amongs all to build up our best model..

'''Imporing few library for create Decision tree classifier and visualizing the tree structure'''

from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score


'''Here we separating independent varibles or target varibles from Iris dataset'''


X = Iris_data[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm','Sepal_petal_wid_len_diff','Sepal_petal_width_diff']]
y = Iris_data['Species']


#Before training the model we have split our data into Actual Train and Actual Test Dataset for training and validating purpose...

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.30, random_state=42)

#spliting data into validation train and validation test
Xt, Xcv, Yt, Ycv = train_test_split(Xtrain, Ytrain, test_size=0.10, random_state=42)


'''Now we have create a Decision tree classifier and trained it with training dataset.'''


Iris_clf = DecisionTreeClassifier(criterion='gini',min_samples_split=2)
Iris_clf.fit(Xt, Yt)

#Visualized the Tree which is formed on train dataset

tree.plot_tree(Iris_clf)


# In[20]:


#Visualizing Decision Tree using graphviz library

dot_data = tree.export_graphviz(Iris_clf, out_file=None)

graph = graphviz.Source(dot_data)
graph


# In[21]:


# As our model has been trained....
#Now we can validate our Decision tree using cross validation method to get the accuracy or performance score of our model.

print('Accuracy score is:',cross_val_score(Iris_clf, Xt, Yt, cv=3, scoring='accuracy').mean())


# In[22]:


#Checking validation test data on our trained model and getting performance metrices

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

Y_hat = Iris_clf.predict(Xcv)


print('Accuracy score for validation test data is:',accuracy_score(Ycv, Y_hat))
multilabel_confusion_matrix(Ycv , Y_hat)


# In[23]:


#Checking our model performance on actual unseen test data.. 
YT_hat = Iris_clf.predict(Xtest)
YT_hat

print('Model Accuracy Score on totally unseen data(Xtest) is:',accuracy_score(Ytest, YT_hat)*100,'%')
multilabel_confusion_matrix(Ytest , YT_hat)


# - As we know our selected feature are working well and model gives very good accuracy score on validate or actual test data. So Now we can trained our model on Actual train dataset with selected features for evaluating/ deploying our model in real world cases. 
# 

# In[24]:


'''Training model on Actual train data... '''
Iris_Fclf = DecisionTreeClassifier(criterion='gini',min_samples_split=2)
Iris_Fclf.fit(Xtrain, Ytrain)

#Visualize tree structure..
tree.plot_tree(Iris_Fclf)


# In[25]:


#Final Decision tree build for deploying in real world cases....

dot_data = tree.export_graphviz(Iris_Fclf, out_file=None)
graph = graphviz.Source(dot_data)
graph


# In[26]:


#Checking the performance of model on Actual Test data...

YT_Fhat = Iris_Fclf.predict(Xtest)
YT_Fhat

print('Model Accuracy Score on totally unseen data(Xtest) is:',accuracy_score(Ytest, YT_Fhat)*100,'%')
multilabel_confusion_matrix(Ytest , YT_Fhat)


# In[27]:


#Testing for New points except from Dataset

Test_point = [[5.4,3.0,4.5,1.5,-1.5,1.5],
             [6.5,2.8,4.6,1.5,-1.8,1.3],
             [5.1,2.5,3.0,1.1,-0.5,1.4],
             [5.1,3.3,1.7,0.5,1.6,2.8],
             [6.0,2.7,5.1,1.6,-2.4,1.1],
             [6.0,2.2,5.0,1.5,-2.8,0.7]]

print(Iris_Fclf.predict(Test_point))

