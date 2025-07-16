import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix




dataset =pd.read_csv('fraudTest.csv')
print(dataset)

dataset.info()
print(dataset.shape)


print("we are try to found how many fraud values are present in ")
dataset['is_fraud'].value_counts()

print("so here we have two values '0' and '1'  which ssuffers from fraud or not ")
print('0- Fraud not  exists/ legitimate transactions    and   1- Fraud  exists/fraud Transactions')

print("ovrall statical calculation of our data ")
dataset.describe()

Lagtr=dataset[dataset['is_fraud']==0]
print("this is lagitimate trasaction ")
Fraudtr=dataset[dataset['is_fraud']==1]
print("this is fraud transaction ")

print(Lagtr.amt.describe())
print(Fraudtr.amt.describe())

numeric_cols = dataset.select_dtypes(include=np.number)
print('select numeric data from dataset')
grouped_means = numeric_cols.groupby(dataset['is_fraud']).mean()
print(grouped_means)

Lagtr.shape
print("lagtr transaction shape",Lagtr.shape)
Fraudtr.shape
print("fraud transaction shape",Fraudtr.shape)
lag_sample= Lagtr.sample(n=2145)
new_dataset= pd.concat([lag_sample,Fraudtr],axis=0)
new_dataset.head()
new_dataset.tail()
new_dataset.shape
new_dataset['is_fraud'].value_counts()
numeric_cols = new_dataset.select_dtypes(include=np.number)
print('select numeric data from dataset')
grouped_means = numeric_cols.groupby(new_dataset['is_fraud']).mean()
print(grouped_means)
x=new_dataset.drop('is_fraud',axis=1)
y= new_dataset['is_fraud']
print(x)
print(y)



x_train ,x_test ,y_train ,y_test =train_test_split(x ,y ,test_size= 0.2 ,stratify=y , random_state=2)

print("X TRAIN DATA IS HERE")
print(x_train)
print()
print("Y TRAIN DATA IS HERE")
print(y_train)
print()
print("X TEST DATA IS HERE")
print(x_test)
print()
print("Y TEST DATA IS HERE")
print(y_test)

print(x.shape, x_train.shape, x_test.shape)
print(y.shape, y_train.shape, y_test.shape)

#LOGESTIC REGRESSION


x_train = pd.DataFrame(x_train).apply(pd.to_numeric, errors='coerce')  
x_test = pd.DataFrame(x_test).apply(pd.to_numeric, errors='coerce')

# 2. Handle missing values (choose one strategy)
x_train = x_train.fillna(x_train.mean())  # Fill with mean
x_test = x_test.fillna(x_train.mean())    # Use training mean for test set


if x_train.ndim == 1:
    x_train = x_train.values.reshape(-1, 1)
if x_test.ndim == 1:
    x_test = x_test.values.reshape(-1, 1)

non_constant_features = [col for col in x_train.columns if x_train[col].nunique() > 1]
x_train = x_train[non_constant_features]
x_test = x_test[non_constant_features]


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("Scaling completed successfully!")
model = LogisticRegression()  
model.fit(x_train_scaled, y_train)
x_train_prod= model.predict(x_train_scaled)
train_data_accuracy = accuracy_score(x_train_prod,y_train)
print("print accuracy on traning data",train_data_accuracy)

print("accuracy on test data")
x_test_pred= model.predict(x_test_scaled)
test_data_accuracy = accuracy_score(x_test_pred,y_test)
print(train_data_accuracy)
 
 
# desision tree
inputs = dataset.drop("is_fraud", axis=1)
inputs
target= dataset['is_fraud']
target
from sklearn.preprocessing import LabelEncoder
le_trans_date_trans_time= LabelEncoder()
le_merchant = LabelEncoder()
le_category	 = LabelEncoder()
le_first	 = LabelEncoder()
le_last = LabelEncoder()
le_gender	 = LabelEncoder()
le_street = LabelEncoder()
le_city = LabelEncoder()
le_state = LabelEncoder()
le_job = LabelEncoder()
le_dob= LabelEncoder()
le_trans_num = LabelEncoder()
inputs['trans_date_trans_time_n'] =le_trans_date_trans_time.fit_transform(inputs['trans_date_trans_time']) 
inputs['merchant_n']=le_merchant.fit_transform(inputs['merchant']) 
inputs['category_n']=le_category.fit_transform(inputs['category']) 
inputs['first_n']=le_first.fit_transform(inputs['first']) 
inputs['last_n']=le_last.fit_transform(inputs['last']) 
inputs['street']=le_street.fit_transform(inputs['street']) 
inputs['city']=le_city.fit_transform(inputs['city']) 
inputs['state']=le_state.fit_transform(inputs['state']) 
inputs['job']=le_job.fit_transform(inputs['job']) 
inputs['dob']=le_dob.fit_transform(inputs['dob']) 
inputs['trans_num']=le_trans_num.fit_transform(inputs['trans_num']) 
inputs

columns_to_drop = ['trans_date_trans_time','merchan','category','first','last',
                   'street','city','state','job','dob','trans_num']
existing_columns = [col for col in columns_to_drop if col in inputs.columns]
inputs_n = inputs.drop(existing_columns, axis='columns')
inputs_n

from sklearn import tree
model= tree.DecisionTreeClassifier()
print(inputs_n.shape)
print(target.shape)

X_train, X_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.3, random_state=42)


X_train = X_train.apply(pd.to_numeric, errors='coerce')  
X_test = X_test.apply(pd.to_numeric, errors='coerce')

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())  

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())


dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42 
)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)  
print("Accuracy on test data:", test_accuracy)
y_train_pred = dt_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Accuracy on training data:", train_accuracy)

