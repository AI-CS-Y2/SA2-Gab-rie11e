import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('cleaned_passwords_data.csv', on_bad_lines='skip')

df = df.dropna(subset=['password', 'strength'])

x = df['password']
y = df['strength']

#transform the passwords into numerical features
vectorizer = CountVectorizer()
x_transformed = vectorizer.fit_transform(x)

#split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.3, random_state=42)

#train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train, y_train)

#predictions on the test set
y_pred = log_reg.predict(x_test)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

#accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy * 100:.2f}%')

#classification report
class_report = classification_report(y_test, y_pred, target_names=['Weak', 'Medium', 'Strong'])
print("\nClassification Report:")
print(class_report)
