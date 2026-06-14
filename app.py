import streamlit as st
import joblib
import numpy as np

model = joblib.load(
    "models/best_model.pkl"
)

vectorizer = joblib.load(
    "models/vectorizer.pkl"
)

feature_names = joblib.load(
    "models/feature_names.pkl"
)

st.title("📰 Fake News Detector")

st.write(
    "Enter any news article below."
)

news = st.text_area(
    "News Content"
)

if st.button("Analyze"):

    vec = vectorizer.transform(
        [news]
    )

    prediction = model.predict(
        vec
    )[0]

    st.divider()

    # Prediction

    if prediction == 0:
        st.error(
            "⚠️ Fake News"
        )
    else:
        st.success(
            "✅ True News"
        )

    # Confidence Score

    if hasattr(
        model,
        "predict_proba"
    ):

        probs = model.predict_proba(
            vec
        )[0]

        st.subheader(
            "Confidence Score"
        )

        st.write(
            f"Fake : {probs[0]*100:.2f}%"
        )

        st.write(
            f"True : {probs[1]*100:.2f}%"
        )

    # Important Keywords

    st.subheader(
        "Detected Keywords"
    )

    indices = vec.nonzero()[1]

    words = []

    for index in indices:

        words.append(
            feature_names[index]
        )

    words = list(
        dict.fromkeys(words)
    )

    for word in words[:10]:

        st.write(
            f"• {word}"
        )