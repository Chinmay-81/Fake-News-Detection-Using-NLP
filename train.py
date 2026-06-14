import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# LOAD DATA
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)

data["content"] = (
    data["title"] + " " + data["text"]
)


# CLEANING
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

data["content"] = data["content"].apply(clean_text)



X_train, X_test, y_train, y_test = train_test_split(
    data["content"],
    data["label"],
    test_size=0.2,
    random_state=42
)


# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names_out()


# MODEL COMPARISON
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000,random_state = 42),
    "Linear SVM": LinearSVC(random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\nMODEL COMPARISON")
print("-" * 40)

for name, model in models.items():

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"{name}: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name


# FINAL REPORT
y_pred = best_model.predict(X_test_vec)

print("\nBest Model:", best_model_name)
print("Accuracy:", best_accuracy)

print(
    classification_report(
        y_test,
        y_pred
    )
)

with open(
    "outputs/model_results.txt",
    "w"
) as f:

    f.write(
        f"Best Model: {best_model_name}\n"
    )

    f.write(
        f"Accuracy: {best_accuracy}\n\n"
    )

    f.write(
        classification_report(
            y_test,
            y_pred
        )
    )

# WORD CLOUD
fake_news_text = " ".join(
    data[data["label"] == 0]["content"]
)

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(fake_news_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")

plt.savefig(
    "outputs/fake_wordcloud.png"
)

plt.close()

# SAVE FILES
joblib.dump(
    best_model,
    "models/best_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)

joblib.dump(
    feature_names,
    "models/feature_names.pkl"
)

print("\nModel Saved Successfully")