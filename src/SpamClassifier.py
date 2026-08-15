import pandas as pd

df = pd.read_csv("data/spam.csv", encoding="latin-1")

print(df.head())
print(df.shape)
print(df.columns)
df = df[['v1', 'v2']]

df.columns = ['label', 'message']

print(df.head())
print(df.columns)
print(df['label'].value_counts())
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['clean_message'] = df['message'].apply(clean_text)

print(df[['message', 'clean_message']].head())
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print(df['label'].head())
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['clean_message'])
y = df['label']

print("TF-IDF shape:", X.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train, y_train)

print("Naive Bayes model trained successfully!")
# Make predictions
y_pred = model.predict(X_test)

print("Predictions completed!")
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Test cases
test_messages = [
    "Congratulations! You have won a free prize. Claim now!",
    "Hey, are you coming to college tomorrow?",
    "You have won a cash reward. Call now to claim it!",
    "Can you send me the assignment notes?",
    "FREE entry to win a brand new phone!"
]

# Clean the test messages
clean_test_messages = [clean_text(message) for message in test_messages]

# Convert messages using the same TF-IDF vectorizer
test_tfidf = vectorizer.transform(clean_test_messages)

# Predict
predictions = model.predict(test_tfidf)

# Display results
print("\n--- Test Case Results ---")

for message, prediction in zip(test_messages, predictions):
    result = "SPAM" if prediction == 1 else "HAM"
    print("\nMessage:", message)
    print("Prediction:", result)