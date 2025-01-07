# Airline Tweets Sentiment Analysis

## Project Overview

The goal of this project is to build a **Sentiment Analysis Model** to classify customer reviews into **positive**, **neutral**, or **negative** sentiments based on tweets about airlines. The project includes data preprocessing, visualization, and model building using various techniques like Bag of Words, TF-IDF, and Word2Vec for feature extraction, followed by classification models for prediction.

---

## Dataset

The dataset used is `Tweets.csv`, which contains airline-related tweets. The relevant fields used in this analysis are:
- `text`: The content of the tweet.
- `airline_sentiment`: The sentiment label, which can be "positive", "neutral", or "negative".
- **Dataset Size**: 14,452 rows.

---

## Steps in the Analysis

### 1. Data Preprocessing
- **Handling Missing Values**: Checked and ensured no null values are present.
- **Removing Duplicates**: Identified and dropped duplicate rows.
- **Noise Removal**: Removed special characters, mentions, and URLs.
- **Stop Words Removal**: Eliminated common stop words using NLTK.
- **Lemmatization**: Reduced words to their root form.
- **Custom Stop Words List**: Created and applied a list to remove additional noise (e.g., "http", "https").

### 2. Exploratory Data Analysis (EDA)
- **Top Tokens**: Extracted and visualized the most frequent words using bar charts.
- **Word Cloud**: Generated a word cloud to depict frequent terms.
- **Sentiment Distribution**: Visualized the distribution of sentiments using bar charts.

### 3. Feature Extraction
- **Bag of Words (CountVectorizer)**: Transformed text into word count vectors.
- **TF-IDF Vectorizer**: Converted text into TF-IDF feature vectors.
- **Word2Vec**: Trained Word2Vec embeddings to capture semantic meaning of words.

### 4. Model Building
- **Logistic Regression**: Built a baseline classification model.
- **Random Forest Classifier**: Improved performance with an ensemble method.

### 5. Model Evaluation
Evaluated models using:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)

---

## Key Results

| Vectorization Method | Model                 | Accuracy | Comments                        |
|----------------------|----------------------|----------|---------------------------------|
| Bag of Words         | Logistic Regression  | 0.7738   | Baseline performance            |
| Bag of Words         | Random Forest        | 0.7527   | Slight improvement over LR      |
| TF-IDF               | Logistic Regression  | 0.7651   | Better feature representation   |
| TF-IDF               | Random Forest        | 0.7534   | Best performance with TF-IDF    |
| Word2Vec             | Random Forest        | 0.6911   | Captures semantic relationships |

### Inferences

1. **Bag of Words vs TF-IDF**: TF-IDF slightly outperforms Bag of Words due to its ability to emphasize important words while reducing the influence of frequently occurring but less significant terms.
2. **Logistic Regression vs Random Forest**: Logistic Regression achieves better accuracy with Bag of Words, likely because it works well with sparse, high-dimensional data. Random Forest performs comparably but does not leverage sparse feature representations as effectively.
3. **Word2Vec**: Accuracy is lower due to the limited dataset size (14,452 rows), which restricts the quality of semantic relationships captured by the embeddings. Pre-trained embeddings like GloVe or FastText could improve this result.

---

## Visualization Examples

### 1. Top 10 Frequent Tokens
![Top Tokens](./images/top_tokens.png)

### 2. Sentiment Distribution
![Sentiment Distribution](./images/sentiment_distribution.png)

### 3. Word Cloud
![Word Cloud](./images/word_cloud.png)

---

## Dependencies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- Gensim
- WordCloud

---

