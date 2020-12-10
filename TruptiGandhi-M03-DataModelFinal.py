# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 19:57:57 2018

@author: Gandhi
"""
'''
 Dataset :Echocardiogram.data

 This is Echocardiogram Dataset. This has published on 28 February 1989 in Miami. The donor of this dataset was Steven Salzberg and collector of this dataset was Dr. Evlin Kinney. Data source is uci.edu.
 
  All the patients suffered heart attacks at some point in the past.
      Some are still alive and some are not.  The survival and still-alive
      variables, when taken together, indicate whether a patient survived
      for at least one year following the heart attack.
 
 What am I going to predict?
 Is patient still alive or dead?
 
 Which categories of machine learning to use?
 1. Supervised learning
 2. Unsupervised learning
 3. Semi- supervised learning
 
 There is an outcome we are trying to predict -  Supervised learning
 
 These are the steps to classify the survival rate of the patient. 
 
 1.Find the data
 2.Apply a data science model
 3.Review the results
 
 '''
 

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

from collections import Counter
from sklearn.metrics import *
import seaborn as sns; sns.set(font_scale=1.2)

#Loading your dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/echocardiogram/echocardiogram.data"

Echocardiogram_df = pd.read_csv(url, header= None, error_bad_lines=False, sep=',')
col = ["survival ", "still-alive", "age-at-heart-attack", "pericardial-effusion","fractional-shortening", "E-point septal separation", "left ventricular end-diastolic", "wall-motion-score","wall-motion-index","mult","name ",  "group ",  "alive-at-1" ]
Echocardiogram_df.columns = col
print ('Check first 5 rows of the table ')
print(Echocardiogram_df.head())

# check the last 5 rows
print ('Check Last 5 rows of the table ')
print(Echocardiogram_df.tail())
# check the total number of rows and column

print ('Print total number of rows and columns ')
print (Echocardiogram_df.shape)

print ('Print name of the columns ')
print(Echocardiogram_df.columns)

Echocardiogram_df.drop(['E-point septal separation'],axis = 1, inplace= True)
print ('Print name of the columns ')
print(Echocardiogram_df.columns)

# remove the space between the words  
# Changing the (' ') into (_) in coulumn names

Echocardiogram_df.columns = Echocardiogram_df.columns.str.strip()
Echocardiogram_df.columns = Echocardiogram_df.columns.str.replace(' ','_')
Echocardiogram_df.columns = Echocardiogram_df.columns.str.replace('-','_')
Echocardiogram_df.columns = Echocardiogram_df.columns.str.replace('\n',' ')
print(Echocardiogram_df.columns)

print ('Check Type of the data ') 
print(Echocardiogram_df.dtypes)


# #checking the missing value per column to decide which column to drop? 

Echocardiogram_df = Echocardiogram_df.replace('[?]', np.NaN, regex = True)
print(Echocardiogram_df.isnull().sum())
Echocardiogram_df.columns.tolist()
Echocardiogram_df.drop(['alive_at_1'],axis= 1, inplace= True)


print(Echocardiogram_df.columns)

# Try to drop the row where 3 columns out of 6 columns have all NAN values. row 32 has 3 NAN values. we are expecting to drop this column.

print(Echocardiogram_df.dropna(subset=['age_at_heart_attack','fractional_shortening','left_ventricular_end_diastolic'],how ='all',inplace =True))

'''
Our dataset has 7 fields for each patient. The first 6 fields describe the health of the patient. These 6 fields are called features. Features are the values that feed into a prediction model. The last field, the still alive, is special. This is the value we are trying to predict. When we use supervised learning to solve a problem, we'll always have the same setup. Features that feed into a supervised learning algorithm which returns one or more target values.
'''

Echocardiogram_df =Echocardiogram_df[['survival', 'age_at_heart_attack','pericardial_effusion','fractional_shortening', 'left_ventricular_end_diastolic', 'wall_motion_index','still_alive']].astype(float)

print ('Check Type of the data ') 
print(Echocardiogram_df.dtypes)
'''
This is a medical data of the petients. I really don't want to assume and add values for missing number. But in this case i have only 130 observation so, I am replacing the NAN and removing the outliers for the purpose of learning. 
'''
Echocardiogram_df ['age_at_heart_attack'] = Echocardiogram_df ['age_at_heart_attack'].astype(float)
Means = np.mean(Echocardiogram_df ['age_at_heart_attack'], axis=0)
print(Means)
Means_round = float(str(round(Means, 2)))
Echocardiogram_df ['age_at_heart_attack']=Echocardiogram_df ['age_at_heart_attack'].replace(np.nan, Means_round)

print(Echocardiogram_df ['fractional_shortening'].dtypes)
Echocardiogram_df ['fractional_shortening'] = Echocardiogram_df ['fractional_shortening'].astype(float)
Median = np.nanmedian(Echocardiogram_df.loc[:,"fractional_shortening"])
print(Median)
IsNan = np.isnan(Echocardiogram_df.loc[:,"fractional_shortening"])
Echocardiogram_df.loc[IsNan,"fractional_shortening"] = Median
Echocardiogram_df ['wall_motion_index'].fillna(method ='ffill', inplace =True)

Echocardiogram_df ['left_ventricular_end_diastolic'] = Echocardiogram_df ['left_ventricular_end_diastolic'].astype(float)
Means = np.mean(Echocardiogram_df ['left_ventricular_end_diastolic'], axis=0)
print(Means)
Means_round = float(str(round(Means, 2)))
Echocardiogram_df ['left_ventricular_end_diastolic']=Echocardiogram_df ['left_ventricular_end_diastolic'].replace(np.nan, Means_round)

Echocardiogram_df ['survival'].fillna(method ='ffill', inplace =True)
# Check the missing value for columns now
print(Echocardiogram_df.isnull().sum())

Echocardiogram_df ['fractional_shortening'].plot('hist')
plt.show()


Echocardiogram_df ['left_ventricular_end_diastolic'].plot('hist')
plt.show()


# An outlier is not necessary a value which stands away from the mean but is a value which was added wrongly to your data. This is a medical report.I am not going to change columns age_at_heart_attack,pericardial_effusion, survival, wall_motion_index
# I am going to remove the outliers from below columns which could have been added wrongly.
# 
# 'FRACTIONAL_SHORTENING' - lower numbers are increasingly abnormal
# 'LEFT_VENTRICULAR_END_DIASTOLIC -Size of heart

def outliers(data):
    q1= np.percentile(data,25)
    q3 = np.percentile(data,75)
    lower = q1-1.5*(q3-q1)
    upper = q1+1.5*(q3-q1)
    flag =(data <= lower)|(data >= upper)
    q2 = np.median(data)
    data[flag] = q2
    return(data)

Echocardiogram_df ['fractional_shortening'] = outliers (Echocardiogram_df ['fractional_shortening'])
Echocardiogram_df ['left_ventricular_end_diastolic']= outliers (Echocardiogram_df ['left_ventricular_end_diastolic'])
Echocardiogram_df.head(25)

x= np.ravel(Echocardiogram_df ['fractional_shortening'])
X= pd.DataFrame(x)
data =  MinMaxScaler().fit_transform(X)
Echocardiogram_df ['fractional_shortening']=data
print(Echocardiogram_df ['fractional_shortening'])

x= np.ravel(Echocardiogram_df ['left_ventricular_end_diastolic'])
X= pd.DataFrame(x)
data =  MinMaxScaler().fit_transform(X)
Echocardiogram_df ['left_ventricular_end_diastolic']=data
print(Echocardiogram_df ['left_ventricular_end_diastolic'])

x= np.ravel(Echocardiogram_df ['wall_motion_index'])
X= pd.DataFrame(x)
data =  MinMaxScaler().fit_transform(X)
Echocardiogram_df ['wall_motion_index']=data
print(Echocardiogram_df ['wall_motion_index'])

Echocardiogram_df.head(25).round(3)

# let's reduce this range appling the binning to get more accuracy on the result.
NB =5
bounds = np.linspace(np.min(x), np.max(x), NB + 1) 
x= np.ravel(Echocardiogram_df ['survival'])
X= pd.DataFrame(x)

bounds = np.linspace(np.min(x), np.max(x), NB + 1)
print (bounds)

def bin(x, b): 
    nb = len(b)
    N = len(x)
    y = np.empty(N, int) 
    
    for i in range(1, nb):
        y[(x >= bounds[i-1])&(x < bounds[i])] = i
    
    y[x == bounds[-1]] = nb - 1
    return y

bx = bin(x, bounds)
print ("\n\nBinned variable x, for ", NB, "bins\n")
print ("Bin boundaries: ", bounds)
print ("Binned variable: ", bx)

Echocardiogram_df['survival'] = bx


NB =5
bounds = np.linspace(np.min(x), np.max(x), NB + 1) 
x= np.ravel(Echocardiogram_df ['age_at_heart_attack'])
X= pd.DataFrame(x)

bounds = np.linspace(np.min(x), np.max(x), NB + 1)
print (bounds)

def bin(x, b): 
    nb = len(b)
    N = len(x)
    y = np.empty(N, int) 
    
    for i in range(1, nb):
        y[(x >= bounds[i-1])&(x < bounds[i])] = i
    
    y[x == bounds[-1]] = nb - 1
    return y

bx = bin(x, bounds)
print ("\n\nBinned variable x, for ", NB, "bins\n")
print ("Bin boundaries: ", bounds)
print ("Binned variable: ", bx)

Echocardiogram_df['age_at_heart_attack'] = bx

#Now we can see from the graph, outliers has been removed from columns 
Echocardiogram_df.head(25).head(3)
Echocardiogram_df.round(3)
'''
# I found my data.Next step is apply data science model.Now In order to **build a model**, the features must be **numeric**, and every observation must have the **same features in the same order**. Let's visualize the data first.to get the more ideas about columns.

# In the below graph I can clearly see that still_alive patients have less fractional_shortening number and high LVED

'''
sns.lmplot('fractional_shortening','left_ventricular_end_diastolic',  data=Echocardiogram_df, hue='still_alive',
           palette='Set1', fit_reg=False, scatter_kws={"s": 30});

'''
In below graph, i can clearly see most the points are collected to the one part of the graph. This is another way to indicate that these columns are playing important role in the model.
'''
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =(Echocardiogram_df ['fractional_shortening'])
y =(Echocardiogram_df ['left_ventricular_end_diastolic'])
z =(Echocardiogram_df ['wall_motion_index'])

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

sns.lmplot('fractional_shortening','left_ventricular_end_diastolic',  data=Echocardiogram_df, hue='still_alive',
           palette='Set1', fit_reg=False, scatter_kws={"s": 30});


# visualize the relationship between the features and the response using scatterplots
sns.set(font_scale=1.5)
sns.pairplot(Echocardiogram_df, hue ="still_alive", palette ='Set1')
plt.show()

'''
 Now 2 part - Apply a data science model
 
 I know that I need to use Supervised learning
 My main Question was "How do I choose which below model to use for my supervised learning task?"
 
 Support Vector Machines
 linear regression
 logistic regression
 naive Bayes
 decision trees
 k-nearest neighbor algorithm
 
 Solution: Model evaluation procedures - It helps to find the best model that represents our data and how well the chosen model will work in the future.
 
 Let's start:
 
 Stage 1: The first Requirement is "Features and response are separate objects" 
 So let's Define the input features and target column
 
'''

#define X and y
X= Echocardiogram_df[['survival','age_at_heart_attack','pericardial_effusion','fractional_shortening','left_ventricular_end_diastolic', 'wall_motion_index']].as_matrix()
y= Echocardiogram_df['still_alive']

#Split the dataset into two pieces: a training set and a testing set.
# Train the model on the training set.
# Test the model on the testing set, and evaluate how well we did.
# split X and y into training and testing sets

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size =.3)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

'''
 How to apply Model?
 
 Step 1. Import the model you want to use
 
 Step 2. Make an instance of the Model
 
 Step 3. Training the model on the data, storing the information learned from the data
 
 Step 4. Predict labels for new data (new images)

 Appling Model -KNN
 Make an instance of Estimator
'''

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
print(knn)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))

#Repeat for KNN with K=1:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
print(knn)
knn.fit(X_train,y_train)


y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))

#Repeat for KNN with K=2:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
print(knn)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))

'''
Accuracy is better with K = 2 so we can use K=2 to predict the target in our dataset. But what if you have big data? For KNN models, complexity is determined by the value of K (lower value = more complex). Let's see if we can use other function to determine the value of K
See if we can locate value for K using function?
'''

# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    


# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# Graph shows that I can pick any number between 13 to 23. let's take K = 15
#Repeat for KNN with K=15:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15)
print(knn)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))


# Create a function to guess when a patient is alive or dead
def Alive_or_Dead(survival,age_at_heart_attack,pericardial_effusion,fractional_shortening,
left_ventricular_end_diastolic,wall_motion_index):
    
    if(knn.predict([[survival,age_at_heart_attack,pericardial_effusion,fractional_shortening,
left_ventricular_end_diastolic,wall_motion_index]]))==0:
        print('You\'re looking at a Alive person!')
    else:
        print('You\'re looking at a Dead person!')
# Predict if person is alive or not
Alive_or_Dead(3,1,.1,.448,.22,.135)

# compare actual response values (y_test) with predicted response values (y_pred)
AR =accuracy_score(y_test, y_pred)
print(AR)


print(confusion_matrix(y_test, y_pred))

CM_log = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = CM_log.ravel()
print(TN, FP, FN, TP )

#Classification Accuracy: Overall, how often is the classifier correct?
print((TP + TN) / float(TP + TN + FP + FN))
print(accuracy_score(y_test, y_pred))


#Classification Error: Overall, how often is the classifier incorrect?

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - accuracy_score(y_test, y_pred))

#Sensitivity: When the actual value is positive, how often is the prediction correct?
#How "sensitive" is the classifier to detecting positive instances?
print(TP / float(TP + FN))
print(recall_score(y_test, y_pred))

#Precision: When a positive value is predicted, how often is the prediction correct?
print(TP / float(TP + FP))
print(precision_score(y_test, y_pred))


# ROC curve can help you to choose a threshold that balances sensitivity and specificity in a way that makes sense for your particular context
print(roc_auc_score(y_test, y_pred))
#f1_score
print(f1_score(y_test, y_pred))

# Store the predicted probabilities for class 1
y_pred_prob = knn.predict_proba(X_test)[:, 1]
print(y_pred_prob)

#ROC curve can help you to choose a threshold that balances sensitivity and specificity in a way that makes sense for your particular context
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(y_test, y_pred_prob) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('ROC curve for Still Alive classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate (1 - Specificity)')
plt.ylabel('TRUE Positive Rate (Sensitivity)')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()


print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(y_test, y_pred_prob), 2), "\n")

# Using a different classification model --Linear regression 

# Linear regression is a simple machine learning method that you can use to predict an observation's value based on the relationship between the target variable and independent, linearily related numeric predictive features.
# import the class
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
# train a logistic regression model on the training set
logreg = LogisticRegression()
# fit the model with data --Model is learning the relationship between X and y
logreg.fit(X_train,y_train)

#  make predictions on the testing set
y_pred = logreg.predict(X_test)
print(y_pred)


# Classification accuracy: percentage of correct predictions
# compare actual response values (y_test) with predicted response values (y_pred)
AR =accuracy_score(y_test, y_pred)
print(AR)

# each row of this matrix corresponds to each one of the classes of the dataset
print ("Coefficients:")
print (logreg.coef_)

# each element of this vector corresponds to each one of the classes of the dataset
print ("Intercept:")
print (logreg.intercept_)


# print the first 15 true and predicted responses
print('True:', y_test[0:15])
print('Pred:', y_pred[0:15])


# Classification accuracy is the easiest classification metric to understand
# but it does not tell you what "types" of errors your classifier is making.
# so we are using the Table that describes the performance of a classification model called as Confusion matrix.

# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
print(confusion_matrix(y_test, y_pred))

CM_log = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = CM_log.ravel()
print(TN, FP, FN, TP)

#Classification Accuracy: Overall, how often is the classifier correct?
print((TP + TN) / float(TP + TN + FP + FN))
print(accuracy_score(y_test, y_pred))

#Classification Error: Overall, how often is the classifier incorrect?
print((FP + FN) / float(TP + TN + FP + FN))
print(1 - accuracy_score(y_test, y_pred))

#Sensitivity: When the actual value is positive, how often is the prediction correct?
#How "sensitive" is the classifier to detecting positive instances?
print(TP / float(TP + FN))
print(recall_score(y_test, y_pred))



#Precision: When a positive value is predicted, how often is the prediction correct?
print(TP / float(TP + FP))
print(precision_score(y_test, y_pred))

#f1_score
print(f1_score(y_test, y_pred))

# Store the predicted probabilities for class 1
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# ROC curve can help you to choose a threshold that balances sensitivity and specificity in a way that makes sense for your particular context

# IMPORTANT: first argument is true values,second argument is predicted probabilities

print(roc_auc_score(y_test, y_pred_prob))

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(y_test, y_pred_prob) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('ROC curve for Still Alive classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate (1 - Specificity)')
plt.ylabel('TRUE Positive Rate (Sensitivity)')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()

####################

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(y_test, y_pred_prob), 2), "\n")


# Create a function to guess when a patient is alive or dead
def Alive_or_Dead(survival,age_at_heart_attack,pericardial_effusion,fractional_shortening,
left_ventricular_end_diastolic,wall_motion_index):
    
    if(logreg.predict([[survival,age_at_heart_attack,pericardial_effusion,fractional_shortening,
left_ventricular_end_diastolic,wall_motion_index]]))==0:
        print('You\'re looking at a Alive person!')
    else:
        print('You\'re looking at a Dead person!')
# Predict if person is alive or not
Alive_or_Dead(3,1,.1,.448,.22,.135)


'''
 define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][1])
    print('Specificity:', 1 - fpr[thresholds > threshold][1])
   '''

from sklearn.svm import SVC, LinearSVC
from sklearn import svm
# Choice of classifier with parameters
t = 0.001 # tolerance parameter
kp = 'rbf' # kernel parameter
clf = svm.SVC(kernel='linear', probability=True)
#clf = SVC(kernel='linear', tol=t, probability=True,C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3)
Still_alive = clf.fit(X_train, y_train)
print(Still_alive)


# Create a function to guess when a patient is alive or dead
def Alive_or_Dead(survival,age_at_heart_attack,pericardial_effusion,fractional_shortening,
left_ventricular_end_diastolic,wall_motion_index):
    
    if(clf .predict([[survival,age_at_heart_attack,pericardial_effusion,fractional_shortening,
left_ventricular_end_diastolic,wall_motion_index]]))==0:
        print('You\'re looking at a Alive person!')
    else:
        print('You\'re looking at a Dead person!')
# Predict if person is alive or not
Alive_or_Dead(3,1,.1,.448,.22,.135)

# make class predictions for X_test_dtm
print ("predictions for test set:")
y_pred = clf.predict(X_test)
print(y_pred)
print ('actual class values:')
print (y_test)

#calculate accuracy of class predictions
AR = accuracy_score(y_test, y_pred)
print(AR)


# each row of this matrix corresponds to each one of the classes of the dataset
print ("Coefficients:")
print (clf.coef_)

# each element of this vector corresponds to each one of the classes of the dataset
print ("Intercept:")
print (clf.intercept_)

# print the first 15 true and predicted responses
print('True:', y_test[0:15])
print('Pred:', y_pred[0:15])

# print the confusion matrix
confusion_matrix(y_test, y_pred)

# print message text for the false negatives (spam incorrectly classified as ham)
print(X_test[y_pred < y_test])

# calculate predicted probabilities for X_test_dtm (poorly calibrated)
y_pred_prob = clf.predict_proba(X_test)[:, 1]
y_pred_prob

#calculate AUC
roc_auc_score(y_test, y_pred_prob)

#Sensitivity: When the actual value is positive, how often is the prediction correct?
#How "sensitive" is the classifier to detecting positive instances?
print(recall_score(y_test, y_pred))


# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(y_test, y_pred_prob) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('ROC curve for Still Alive classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate (1 - Specificity)')
plt.ylabel('TRUE Positive Rate (Sensitivity)')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()

####################

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(y_test, y_pred_prob), 2), "\n")

'''
 Let's collect the data and compare their results now. 
  
 True positive: Still alive people correctly identified as Still alive 
  False positive: Dead people incorrectly identified as Alive 
  True negative: Dead people correctly identified as dead
  IMP : False negative: Alive people incorrectly identified as Dead
  In medical field,
 Still alive people incorrectly identified as dead is not acceptable. False Negative value for KNN and LR is 3 and for SVC is 5
  Because false positives ( Dead people incorrectly identified as Alive) are more acceptable than false negatives ( Alive people incorrectly identified as Dead)
  sensitivity is important in this case.
 
 sensitivity for KNN is .66
 sensitivity for LR is .66
 Asensitivity for SVC is .44
 
 Now I will check the accuracy.
 
 I will choose the KNN model in this case.
'''
# 
# 





















