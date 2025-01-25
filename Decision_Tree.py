import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#load the dataset
df = pd.read_csv('cleaned_passwords_data.csv', on_bad_lines='skip')

#drop missing values
df = df.dropna(subset=['password', 'strength'])

x = df['password']
y = df['strength']

#vectorize the passwords using CountVectorizer
vectorizer = CountVectorizer()
x_transformed = vectorizer.fit_transform(x)

#split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.3, random_state=42)

tree_clf = DecisionTreeClassifier(max_depth=10, random_state=42)

x_train_small = x_train[:15000]
y_train_small = y_train[:15000]

tree_clf.fit(x_train_small, y_train_small)

#predictions on the test set
y_pred_tree = tree_clf.predict(x_test)

#confusion matrix
cm = confusion_matrix(y_test, y_pred_tree)
print("Decision Tree Confusion Matrix:")
print(cm)

#accuracy score
accuracy = accuracy_score(y_test, y_pred_tree)
print(f'Decision Tree Accuracy: {accuracy * 100:.2f}%')
