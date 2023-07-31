import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset - Replace with your own dataset
data = pd.read_csv('fake_news_dataset.csv')
X = data['text']
y = data['label']

# Convert labels to binary values
y = y.map({'REAL': 0, 'FAKE': 1})

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the text data to numerical feature vectors
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize PassiveAggressiveClassifier
clf = PassiveAggressiveClassifier(max_iter=50)

# Train the model
clf.fit(tfidf_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(tfidf_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{confusion_mat}")

# Print the classification report
class_report = classification_report(y_test, y_pred, target_names=['REAL', 'FAKE'])
print("Classification Report:")
print(class_report)

# Get the most important features (words) for fake news detection
feature_names = tfidf_vectorizer.get_feature_names_out()
top_fake_indices = clf.coef_[0].argsort()[-10:]
print("Top 10 words associated with fake news:")
for idx in top_fake_indices:
    print(f"{feature_names[idx]}")

# Get the least important features (words) for fake news detection
bottom_fake_indices = clf.coef_[0].argsort()[:10]
print("Top 10 words associated with real news:")
for idx in bottom_fake_indices:
    print(f"{feature_names[idx]}")
