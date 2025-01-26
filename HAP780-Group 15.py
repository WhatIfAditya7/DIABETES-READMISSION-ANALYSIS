#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
data=pd.read_csv("C:/Users/adish/OneDrive/Desktop/HAP780/Project/readmission diabetes.csv")
#Read the file


# In[29]:


print(data)


# In[30]:


frequency = data['race'].value_counts().reset_index()
frequency.columns = ['Race', 'Frequency']
print(frequency)
#To check the frequency of variables


# In[31]:


sh=data.shape
print(sh)
#Shape of the dataset


# In[32]:


import numpy as np
data.replace('?',np.nan,inplace=True)
print(data)
#Replacing the ? with NA 


# In[33]:


nan_per_column = data.isna().sum()
print(nan_per_column)
#Checking NA per columns


# In[34]:


#data.drop(columns=['max_glu_serum', 'A1Cresult'], inplace=True)
data.drop(columns=['weight', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult',
                  'age','examide','citoglipton','encounter_id','patient_nbr',
                               'admission_type_id','discharge_disposition_id','admission_source_id',
                               'number_emergency','diag_1','diag_2','diag_3','num_medications',
                               'number_outpatient','number_inpatient','number_diagnoses'], inplace=True)
print(data)
#Removing unwanted columns 


# In[35]:


sh=data.shape
print(sh)
nan_per_column = data.isna().sum()
print(nan_per_column)


# In[36]:


cleaned_data = data.dropna()
print(cleaned_data)


# In[40]:


cleaned_data['readmitted']=cleaned_data['readmitted'].replace({'>30':'Admitted after 30 days','<30':'Admitted within 30 days'})
print(cleaned_data)
#Mapping the independent variable


# In[88]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.countplot(x='race', hue='time_in_hospital', data=cleaned_data, palette='mako')
plt.title('Average Age by Race', fontsize=16)
plt.xlabel('Race', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[43]:


plt.figure(figsize=(10, 6))
sns.countplot(x='race', hue='gender', data=cleaned_data, palette='cubehelix')
plt.title('Age vs. Race', fontsize=16)
plt.xlabel('Race', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[44]:


plt.figure(figsize=(10, 6))
sns.countplot(x='gender', hue='readmitted', data=cleaned_data, palette='rocket')
plt.title('Readmission Rate vs. Gender', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[45]:


plt.figure(figsize=(10, 6))
sns.countplot(x='race', hue='readmitted', data=cleaned_data, palette='viridis')
plt.title('Readmission Rate vs. Race', fontsize=16)
plt.xlabel('Race', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[47]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
data_en=pd.get_dummies(cleaned_data,columns=['race','gender','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
                                           'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
                                           'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
                                           'insulin','glyburide-metformin','glipizide-metformin',
                                           'glimepiride-pioglitazone','metformin-rosiglitazone',
                                           'metformin-pioglitazone','change','diabetesMed'])
print(data_en)
#Initializing the ONEHOTENCODING 


# In[48]:


data_en = data_en.applymap(lambda x: 1 if x is True else 0 if x is False else x)
print(data_en)
#Mapping the 0-1 with True-False


# In[68]:


from sklearn.model_selection import train_test_split
x=data_en.drop('readmitted',axis=1)
y=data_en['readmitted']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#Splitting the data into 80% training and 20% testing


# In[91]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
mapping1 = {'NO': 0, 'Admitted within 30 days': 1, 'Admitted after 30 days': 2}
y_encoded = y.map(mapping1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
#INnitializng the PCA


# In[102]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
m1 = DecisionTreeClassifier(random_state=42,max_depth=5, min_samples_split=3, min_samples_leaf=3)
m1.fit(X_train_pca, y_train)
y_pred = m1.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
#J48 Modelling


# In[106]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(60,15))  # Adjust the size of the plot as needed
plot_tree(m1, filled=True, feature_names=x.columns.tolist(), class_names=['No', 'Yes','Maybe'],fontsize=10, rounded=True)
plt.show()


# In[80]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_pca, y_train)
y_pred = rf_model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
#Random Forest Modelling


# In[112]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model = LogisticRegression(
    random_state=42,
    multi_class='multinomial',  # Specify multinomial logistic regression
    solver='lbfgs',             # Recommended solver for multinomial problems
    max_iter=1000               # Increase iterations if needed
)

model.fit(X_train_pca, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#MultiNominal Logsitic Regresison


# In[85]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have feature importances and feature names in lists or arrays
feature_importances = m1.feature_importances_  # Replace 'm1' with your DecisionTreeClassifier variable
feature_names = x_pca_d1.columns  # Replace with your DataFrame containing feature names

# Create a DataFrame for better visualization and sorting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort the DataFrame by importance values in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances from Decision Tree')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()
#Plotting the important features


# In[107]:


plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance per Principal Component')
plt.show()


# In[109]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)  # Use (C - 1) components, where C = number of classes
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train_lda, y_train)

# Predictions and Evaluation
y_pred = clf.predict(X_test_lda)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#Uisng the PLS and applying the J48.

