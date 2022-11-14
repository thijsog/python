# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:22:17 2022

@author: thijs
"""
#%% ---### Loading in data and setting up the DF ###---
## load in package ##
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import os

# model training
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

## set working directory ##
os.chdir("C:/Users/thijs/OneDrive/Documenten/python oefen folder")

#%%% ## creating a dataframe from pandas series ##
data = {'State': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
 'Year': [2000, 2001, 2002, 2001, 2002, 2003],
 'Population': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
df = pd.DataFrame(data)
df

## creating a dataframe using data stored in a csv file ##

df = pd.read_csv(filepath_or_buffer = 'movies.csv', delimiter=',', 
                 doublequote=True, quotechar='"',na_values = ['na', '-', '.', ''], 
                 quoting=csv.QUOTE_ALL, encoding = "ISO-8859-1") # double quote is om rekening te houden "" in columns, encoding is om data op te slaan als bytes
df

# werkt ook
df_test = pd.read_csv(filepath_or_buffer = 'movies.csv', delimiter=',', 
                na_values = ['na', '-', '.', ''], 
                  encoding = "ISO-8859-1")
df_test

#%% ---### exploration ###--- [PART 1]
df.info() # displays the index and the data types
df.head(5) # displays the first 5 rows

df.iloc[:, 5:10] # displays columns 6 till 10 and all their rows
df.iloc[1:5,1:3] # diplays the first 4 rows of the 2nd and 3th column

#%%% ---## select specific parts of the data ##---
df.loc[(df['content_rating'] == 'PG-13').values, ['actor_1_facebook_likes', 'actor_3_facebook_likes', 'budget']]

df.loc[df['content_rating'] == 'PG-13', ['actor_1_facebook_likes', 'actor_3_facebook_likes', 'budget']]

df['content_rating'] == 'PG-13'

# ook nog een optie 
cols_set2 = df[df.columns[[5,7,9]]][:]
cols_set2

# maak je er echt een apart object van
df2 = df.loc[(df['content_rating'] == 'PG-13').values, ['actor_1_facebook_likes', 'actor_3_facebook_likes', 'budget']]

df3 = df.loc[df['content_rating'] == 'PG-13', ['actor_1_facebook_likes', 'actor_3_facebook_likes', 'budget']]

# maar dan cijfer based
df4 = df.loc[df.iloc[:,21] == 'PG-13', ['actor_1_facebook_likes', 'actor_3_facebook_likes', 'budget']]

df.iloc[:,21]


# uitgebreider > twee selectie criteria
df.loc[(df.iloc[:,21] == 'PG-13') & (df.iloc[:,22] <= 200000), ['actor_1_facebook_likes', 'actor_3_facebook_likes', 'budget']]


#%%% ---## profiling the table (printing the min, max and average of the attributes witth numerical values) #---#
dataTypeSeries = df.dtypes
for col_idx in range(len(df.columns)):
       if (not (dataTypeSeries[col_idx] == 'object')):
            print(df.columns[col_idx], 'has Min = ', df[df.columns[col_idx]].min(), 
                  'Max = ', df[df.columns[col_idx]].max(), 
                  'Average = ', df[df.columns[col_idx]].mean())

range(len(df.columns))

for x in range(len(df.columns)):
  print(x)
  
  
df.iloc[:,0] # 0 is de eerste column

#%%% ---## removing missing values ##---
# use a Python function to delete the records with missing values
idf_copy = df.copy()
idf_copy.dropna()

# replace the missing values with the following value
idf_copy = df.copy()
idf_copy.fillna(-99)


#%%  ---### examples from ADS assignment ###---
df_ads = pd.read_csv('fdExample.csv')

# list of unique first names of males
male = df_ads[df_ads['Gender'] == 'M']['First name'].unique() # dus conditie waar rows aan moeten voldoen en dan daarna de column die moet worden returnd.

df_ads[(df_ads["Gender"] == "M") & (df_ads["Department"] == "COR")]["Department"].count() # amount of males in de COR department

## kan je een loop maken voor alle departmens en dan staafdiagrammen?


## eerst empty df
test_df = pd.DataFrame(data=None,columns = ['Gender','Department', 'Amount_empl'])

for col_test in df_ads.columns:
        test_df["Gender"].append("M")
        test_df["Gender"].append("F")
        
        test_df["Department"].append(col_test)
        test_df["Department"].append(col_test)
        
        test_df['Amount_empl'].append(df_ads[(df_ads["Gender"] == "M") & (df_ads["Department"] == col_test)]["Department"].count())
        test_df['Amount_empl'].append(df_ads[(df_ads["Gender"] == "F") & (df_ads["Department"] == col_test)]["Department"].count())

# dit werkt maar is niet mooi
G = []
D = []
AMPL = []

for col_test in df_ads["Department"].unique():
    G.append("M")
    G.append("F")
    D.append(col_test)
    D.append(col_test)
    AMPL.append(df_ads[(df_ads["Gender"] == "M") & (df_ads["Department"] == col_test)]["Department"].count())
    AMPL.append(df_ads[(df_ads["Gender"] == "F") & (df_ads["Department"] == col_test)]["Department"].count())

test_df = pd.DataFrame()
test_df["Gender"] = G
test_df["Department"] = D
test_df["Amount_empl"] = AMPL 


## barplot ##
X_axis = np.arange(len(df_ads["Department"].unique()))

plt.bar(X_axis - 0.2,test_df[test_df["Gender"] == "M"]["Amount_empl"], 0.4, label = "Male")
plt.bar(X_axis + 0.2,test_df[test_df["Gender"] == "F"]["Amount_empl"], 0.4, label = "Female")

plt.xticks(X_axis, df_ads["Department"].unique(),rotation = 90)
plt.subplots_adjust(bottom = 0.200)
plt.xlabel("Departments")
plt.ylabel("Number of employees")
plt.title("Number of employees in each department")
plt.tight_layout(rect=[0, 0, 2, 1]) # dit zorgt er voor dat het past 
plt.legend()
plt.show()

#%% ## ---- training a model ---- ## [PART 2]

# load in the dataset
df_dia = pd.read_csv('diabetes.csv')

# split the dataset and stratisfy 
X_train, X_test, y_train, y_test = train_test_split(df_dia.iloc[: , :8], df_dia["Outcome"], test_size = 0.25, 
                                                    shuffle = True,
                                                    stratify = df_dia["Outcome"]) # stratify als er groepen zijn (positief en negatief), iloc is voor selectie zonder uitkomst

#%%% Logistic regression classifier
# Train the model with the training subset
logreg = LogisticRegression(max_iter= 200) # increased it to 200 in order to prevent convergence warning.
logreg.fit(X_train, y_train)

# Test the model on the test subset
result_lg = logreg.predict(X_test)

# display the confusion matrix
confusion_matrix(result_lg, y_test)

# display the performaance of the Logistic regression model.
f1_lg = f1_score(y_test, result_lg, average=None)
acc_lg = accuracy_score(y_test, result_lg, normalize=True)
bacc_lg = balanced_accuracy_score(y_test, result_lg)
print ("F1-Score = {}\nAccuracy = {}\nBalanced Accuracy = {}".format(f1_lg, acc_lg, bacc_lg))


df_dia.iloc[: , :8]
#%%% random forrest classifier
## train the random forrest classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Test the model on the test subset
result_rfc = rfc.predict(X_test)

# display the confusion matrix
confusion_matrix(result_rfc, y_test)

df_dia["Outcome"][df_dia["Outcome"] == 1].count()
df_dia["Outcome"][df_dia["Outcome"] == 0].count()

#!! check dit nog even  > https://note.nkmk.me/en/python-multi-variables-values/
unique,counts = np.unique(result_lg,return_counts=True)

test2 = np.unique(result_lg,return_counts=True) # als je het zo doet krijg je een tupple > twee vars is handig want dan staat het los.


# display the performaance of the Random forrest model.
f1_rfc = f1_score(y_test, result_rfc, average=None)
acc_rfc = accuracy_score(y_test, result_rfc, normalize=True)
bacc_rfc = balanced_accuracy_score(y_test, result_rfc)
print ("F1-Score = {}\nAccuracy = {}\nBalanced Accuracy = {}".format(f1_rfc, acc_rfc, bacc_rfc))


#%%% SVM classifier
# Train the model with the training subset
SVM = svm.SVC()
SVM.fit(X_train, y_train)

# Test the model on the test subset
result_svm = SVM.predict(X_test)

# display the confusion matrix
confusion_matrix(result_svm, y_test)

# display the performaance of the SVM model.
f1_svm = f1_score(y_test, result_svm, average=None) # acc voor 0 en dan 1
acc_svm = accuracy_score(y_test, result_svm, normalize=True) # real accuracy 
bacc_svm = balanced_accuracy_score(y_test, result_svm) # accuracy for imbalanced datasets
print ("F1-Score = {}\nAccuracy = {}\nBalanced Accuracy = {}".format(f1_svm, acc_svm, bacc_svm)) 