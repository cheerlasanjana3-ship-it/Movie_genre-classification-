# Movie_genre-classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("movies.csv")

# Input and output
X = df["plot"]
y = df["genre"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numbers
tfidf = TfidfVectorizer(stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = LinearSVC()

model.fit(X_train_tfidf, y_train)

# Test model
y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred))

# Try custom input
new_plot = ["A brave police officer fights crime and corruption in the city"]
new_plot_tfidf = tfidf.transform(new_plot)
print("Predicted genre:", model.predict(new_plot_tfidf))

