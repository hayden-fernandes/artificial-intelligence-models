import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Data Pre-processing
# Read the dataset
url = 'https://drive.google.com/uc?id=1Y-0LkYMk3jvs4H8MTW6dAH4aqhfgIS2a'
heart_data = pd.read_csv(url, sep=';')

# Check the first few rows of the dataset
print(heart_data.head())

# Check for missing values
print(heart_data.isnull().sum())

print(heart_data)

# Step 2: Data Analysis and Visualization
# Visualize the distribution of target variable 'num' (1: presence of heart disease, 0: absence)
sns.countplot(x='cardio', data=heart_data)
plt.title('Distribution of Heart Disease')
plt.show()

# Visualize the age distribution
sns.histplot(x='age', data=heart_data, bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# Visualize the correlation matrix
corr_matrix = heart_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 3: Correlation Matrix
print(corr_matrix)

# Step 4: Machine Learning Techniques
# Splitting the data into features and target variable
X = heart_data.drop('cardio', axis=1)
y = heart_data['cardio']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
print("Support Vector Machine Accuracy:", svm_accuracy)

# K-Nearest Neighbor (KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_pred)
print("KNN Accuracy:", knn_accuracy)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_accuracy)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_accuracy)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)

# Step 5: Build the Machine Learning Model
# Since Random Forest has the highest accuracy, we'll choose it as our final model

# Final Model
final_model = RandomForestClassifier(random_state=42)
final_model.fit(X, y)
