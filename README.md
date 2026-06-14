# Fake News Detection using Machine Learning and NLP

## Overview

This project detects whether a news article is **Fake News** or **True News** using Natural Language Processing (NLP) and Machine Learning techniques.
The system analyzes news content, converts text into numerical features using TF-IDF, and applies machine learning algorithms to classify the article.
A Streamlit web application is included for real-time prediction.

---

## Features

* News Classification (Fake / True)
* Text Cleaning and Preprocessing
* TF-IDF Feature Extraction
* Multiple Model Comparison

  * Multinomial Naive Bayes
  * Logistic Regression
  * Linear SVM
* Automatic Best Model Selection
* Confidence Score Display
* Important Keyword Extraction
* Word Cloud Visualization
* Streamlit Web Application
* Model Persistence using Joblib

---

## Dataset

The project uses the Fake and True News dataset from Kaggle.

Dataset contains:

* Fake News Articles
* True News Articles

Total Articles: 44,000+

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Streamlit
* Joblib
* Matplotlib
* WordCloud

---

## Project Structure
```
Fake-News-Detection/
├── data/
│   ├── Fake.csv
│   └── True.csv
|
├── models/
│    ├── best_model.pkl
│    ├── vectorizer.pkl
│    └── feature_names.pkl
|
├── outputs/
│   ├── fake_wordcloud.png
│   └── model_results.txt
|
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Fake-News-Detection.git

cd Fake-News-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run:

```bash
python train.py
```

This will:

* Train multiple models
* Compare performance
* Select the best model
* Save trained artifacts
* Generate a word cloud

---

## Run Prediction Script

```bash
python predict.py
```

---

## Run Web Application

```bash
streamlit run app.py
```

---

## Example Output

Input:

Breaking News: Government releases new election report.

Output:

TRUE NEWS

Confidence Score:

True: 97.5%

Fake: 2.5%

Detected Keywords:

* government
* election
* report
* releases

---

## Future Improvements

* BERT-Based Fake News Detection
* Explainable AI (SHAP/LIME)
* Real-Time News Verification
* REST API Deployment
* AWS Cloud Deployment

---

## Author

Chinmay nayak

Machine Learning | NLP | Streamlit
