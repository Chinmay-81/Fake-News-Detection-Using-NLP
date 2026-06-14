import joblib

model = joblib.load(
    "models/best_model.pkl"
)

vectorizer = joblib.load(
    "models/vectorizer.pkl"
)

while True:

    news = input(
        "\nEnter News Article:\n"
    )

    vec = vectorizer.transform(
        [news]
    )

    prediction = model.predict(
        vec
    )[0]

    if prediction == 0:
        print("\nFAKE NEWS")
    else:
        print("\nTRUE NEWS")