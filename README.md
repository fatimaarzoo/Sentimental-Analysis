# Sentiment Analysis Using Machine Learning

A machine learning project for multi-class sentiment analysis using TF-IDF feature extraction and multiple classification algorithms.

## 📌 Project Overview

This project analyzes sentiment from social-media-style text using Natural Language Processing (NLP) and machine learning.

### Project Workflow

- Dataset loading and exploration
- Exploratory Data Analysis (EDA)
- Text cleaning and preprocessing
- TF-IDF feature extraction
- Training multiple ML models
- Model comparison and evaluation
- Saving the trained model
- Predicting sentiment for new text
- Exporting the cleaned dataset

## 📂 Dataset

The project uses `sentimentdataset.csv`.

- Records: 732
- Columns: 15
- Text column: `Text`
- Target column: `Sentiment`

### Important Columns

| Column | Description |
|---|---|
| `Text` | Original text |
| `Sentiment` | Sentiment/emotion label |
| `Timestamp` | Date and time |
| `Platform` | Social media platform |
| `Hashtags` | Associated hashtags |
| `Retweets` | Number of retweets |
| `Likes` | Number of likes |
| `Country` | Country associated with the post |

Rare sentiment classes containing fewer than two samples are grouped into an `Other` category.

## 🛠️ Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- WordCloud
- NLTK
- Scikit-learn
- Joblib

## 🔄 Project Workflow

Dataset → Data Exploration → Text Cleaning → TF-IDF → Train/Test Split → Model Training → Evaluation → Prediction

## 📊 Exploratory Data Analysis

The project performs EDA to understand the dataset.

### Sentiment Distribution

The top 10 most frequent sentiment categories are visualized.

### Platform-wise Sentiment

Sentiment distributions are compared across different platforms.

### Country-wise Sentiment

The distribution of major sentiments across countries is analyzed.

### Likes vs Retweets

A scatter plot is used to explore the relationship between likes and retweets.

### Sentiment Activity by Hour

Sentiment activity is analyzed across different hours of the day.

## 🧹 Text Preprocessing

The text cleaning pipeline:

- Converts text to lowercase
- Removes URLs
- Removes user mentions
- Removes hashtags
- Removes non-alphabetic characters
- Removes English stop words
- Removes unnecessary whitespace

### Example

Original:

`Enjoying a beautiful day at the park!`

Cleaned:

`enjoying beautiful day park`

The cleaned text is stored in `df["Clean_Text"]`.

## 🔢 TF-IDF Feature Extraction

TF-IDF is used to convert text into numerical features.

### Configuration

- Maximum features: 15,000
- N-grams: Unigrams, bigrams, and trigrams
- Minimum document frequency: 2
- Maximum document frequency: 95%

### Feature Matrix

- Training samples: 585
- Testing samples: 147
- TF-IDF features: 1,758

The dataset is split using stratification with `random_state=42`.

## 🤖 Machine Learning Models

Four classification algorithms are trained and compared.

### Logistic Regression

`LogisticRegression(max_iter=200)`

### Multinomial Naive Bayes

`MultinomialNB()`

### Linear SVM

`LinearSVC()`

### Random Forest

`RandomForestClassifier(n_estimators=200, random_state=42)`

## 📈 Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 21.77% |
| Multinomial Naive Bayes | 19.73% |
| Linear SVM | 46.26% |
| Random Forest | 43.54% |

Linear SVM achieved 46.26% accuracy on the held-out test set.

### Detailed Evaluation

| Metric | Score |
|---|---:|
| Accuracy | ~46% |
| Macro F1 | ~0.37 |
| Weighted F1 | ~0.42 |

The relatively low performance is mainly associated with the large number of sentiment categories and the small number of examples available for many classes.

## 🔮 Sentiment Prediction

A prediction function cleans new text, transforms it using the trained TF-IDF vectorizer, and predicts its sentiment.

### Example

`predict_sentiment("I hate this movie!")`

### Example Predictions

| Input | Prediction |
|---|---|
| I hate this movie! | Hate |
| I love the experience so much | Love |
| I am so glad I came today | Negative |
| The service was awful and slow | Disappointed |
| I cried cause it was that bad | Bad |

Predictions should be treated as model outputs because the dataset contains many fragmented sentiment categories.

## 💾 Saving the Model

The trained model and TF-IDF vectorizer are saved using Joblib.


## 🚀 Future Improvements

- Combine similar sentiment categories
- Create broader Positive/Negative/Neutral classes
- Increase training data
- Apply class-balancing techniques
- Perform hyperparameter tuning
- Use stemming or lemmatization
- Test character n-grams
- Use BERT or other transformer models
- Add cross-validation
- Build a Streamlit web application



⭐ If you found this project useful, consider starring the repository!
