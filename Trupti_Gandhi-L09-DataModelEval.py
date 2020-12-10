# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 19:57:57 2018

@author: Gandhi
"""
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

#Loading your dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/echocardiogram/echocardiogram.data"

Echocardiogram_df = pd.read_csv(url, header= None, error_bad_lines=False, sep=',')
col = ["survival ", "still-alive", "age-at-heart-attack", "pericardial-effusion","fractional-shortening", "E-point septal separation", "left ventricular end-diastolic", "wall-motion-score","wall-motion-index","mult","name ",  "group ",  "alive-at-1" ]
Echocardiogram_df.columns = col
print ('Check first 5 rows of the table ')
print(Echocardiogram_df.head())


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

Echocardiogram_df =Echocardiogram_df[['survival', 'age_at_heart_attack','pericardial_effusion','fractional_shortening', 'left_ventricular_end_diastolic', 'wall_motion_index','still_alive']].astype(float)

print ('Check Type of the data ') 
print(Echocardiogram_df.dtypes)


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
print(Echocardiogram_df.isnull().sum())

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


Echocardiogram_df.head(25).head(3)
Echocardiogram_df.round(3)

# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns; 
sns.set(font_scale=1.2)

# Allows charts to appear in the notebook
get_ipython().magic('matplotlib inline')

sns.lmplot('fractional_shortening','left_ventricular_end_diastolic',  data=Echocardiogram_df, hue='still_alive',
           palette='Set1', fit_reg=False, scatter_kws={"s": 30});


# visualize the relationship between the features and the response using scatterplots
sns.set(font_scale=1.5)
sns.pairplot(Echocardiogram_df, hue ="still_alive", palette ='Set1')
plt.show()

#define X and y
X= Echocardiogram_df[['survival','age_at_heart_attack','pericardial_effusion','fractional_shortening','left_ventricular_end_diastolic', 'wall_motion_index']].as_matrix()
y= Echocardiogram_df['still_alive']

#Split the dataset into two pieces: a training set and a testing set.
# Train the model on the training set.
# Test the model on the testing set, and evaluate how well we did.

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size =.3)

print(X_train.shape)

print(X_test.shape)
print(y_train.shape)

print(y_test.shape)


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

# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    
    # import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().magic('matplotlib inline')

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


print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(y_test, y_pred_prob), 2), "\n")





















