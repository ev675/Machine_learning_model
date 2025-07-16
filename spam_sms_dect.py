import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

data = pd.read_csv('spam (2).csv')
print(data)
print(data.head(10))
print(data.info())
print(data.shape)
print(data['v1'].value_counts())
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data.loc[data['v1']== 'spam','v1']=0
data.loc[data['v1']== 'ham','v1']=1
print(data['v1'].value_counts())
data.isnull().sum()
dele =['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
exist =[colmn for colmn in dele if colmn in data.columns]
data.drop(columns=exist,inplace= True)
print(data)
x= data['v2']
y= data['v1']
print(x)
print(y)
x_train, x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)
feature_extraction = TfidfVectorizer(min_df=1 , stop_words='english',lowercase= True)
x_train_feature = feature_extraction.fit_transform(x_train)
y_train_feature = feature_extraction.fit_transform(x_test)

y_train= y_train.astype('int')
y_test= y_test.astype('int')

print(x_train_feature[:1])
print(y_train_feature[:1])
model = LogisticRegression()
model.fit(x_train_feature ,y_train )
model.fit(y_train_feature,y_test)
train_pre =model.predict(x_train_feature)
print(train_pre.shape)

from sklearn.metrics import recall_score
print('Accuracy on training data:', accuracy_score(y_train,train_pre))
print('Confusion matrix :\n',confusion_matrix(y_train,train_pre))
print('precision: ', precision_score(y_train,train_pre))
print('recall:', recall_score(y_train,train_pre))

test_pre =model.predict(y_train_feature)
test_pre
from sklearn.metrics import recall_score
print('Accuracy on training data:', accuracy_score(y_test,test_pre))
print('Confusion matrix :\n',confusion_matrix(y_test,test_pre))
print('precision: ', precision_score(y_test,test_pre))
print('recall:', recall_score(y_test,test_pre))
print(x[:1])
input_data =input('enter a sms')
input_data_features= feature_extraction.transform([input_data])
prediction = model.predict(input_data_features)
print(prediction)
if(prediction[0]==1):
    print('Ham sms ')
else:
    print('Spam sms')

import pickle
with open('spam_sms_dect.pkl','wb')as file:
    pickle.dump(model,file)    
