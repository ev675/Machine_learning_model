import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
set= pd.read_csv('Churn_Modelling.csv')
set
set.dtypes
set.info()
set.describe()
set.duplicated()
set.describe().sum()
set.values
set.shape
set.isnull().sum()
for col in set.columns:
    print(col)
    print(set[col].unique())
    print("")
set['Exited'].value_counts()
set['Gender'].value_counts()
set['Geography'].value_counts()
set.replace({"Geography": {'France': 0, 'Germany': 1, 'Spain': 2}}, inplace=True)
set['Geography'] = set['Geography'].astype(int)  
cols_to_drop = ['RowNumber', 'Surname']
missing_cols = [col for col in cols_to_drop if col not in set.columns]
if not missing_cols:
    set.drop(columns=cols_to_drop, inplace=True)
else:
    print(f"Columns not found: {missing_cols}")

print(set)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


y = set['Exited']
x = set.drop('Exited', axis=1)
x = pd.get_dummies(x, dtype=int)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


logreg = LogisticRegression(max_iter=1000)  
logreg.fit(x_train_scaled, y_train)


train_pred = logreg.predict(x_test_scaled)


print("Test Accuracy:", accuracy_score(y_test, train_pred))
print("\nClassification Report:")
print(classification_report(y_test, train_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, train_pred))


logreg.fit(x_test_scaled, y_test )
test_pred = logreg.predict(x_test_scaled)

print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("\nClassification Report:")
print(classification_report(y_test, test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Prepare the data (same as before)
y = set['Exited']
x = set.drop('Exited', axis=1)
x = pd.get_dummies(x, dtype=int)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


rf = RandomForestClassifier(
    n_estimators=100,  
    max_depth=5,       
    random_state=42,   
    class_weight='balanced'  
)

rf.fit(x_train, y_train)


y_pred = rf.predict(x_test)
y_pred_proba = rf.predict_proba(x_test)[:, 1]  

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


feature_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance Score')
plt.title('Feature Importance')
plt.show()
    
