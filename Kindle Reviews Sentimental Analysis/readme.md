# Sentiment Analysis on Amazon Kindle Reviews

This project focuses on building machine learning models to perform sentiment analysis on Amazon Kindle Store reviews. The dataset used contains product reviews with their associated ratings, and the goal is to classify reviews as either positive (rating >= 3) or negative (rating < 3).

## About the Dataset

**Context**:
This is a small subset of the dataset of book reviews from the Amazon Kindle Store category. The dataset spans from May 1996 to July 2014 and contains 982,619 entries. Each reviewer has at least 5 reviews, and each product has at least 5 reviews.

**Dataset Used**:
A filtered version containing 12,000 reviews was used for this analysis.

## Project Workflow

### 1. Data Loading and Preprocessing
- The dataset was loaded using `pandas`.
- Missing values were checked and handled.
- Reviews were converted to lowercase, special characters and URLs were removed, and stopwords were filtered out using NLTK.
- Lemmatization was applied to normalize the text.

### 2. Feature Engineering
#### Bag of Words (BoW)
- Features were extracted using `CountVectorizer`.
- Training data was transformed using `fit_transform` and test data using `transform` to prevent data leakage.

#### TF-IDF (Term Frequency-Inverse Document Frequency)
- Features were extracted using `TfidfVectorizer` with similar preprocessing steps as BoW.

#### Word2Vec
- **Word Embeddings**: Trained a `Word2Vec` model from scratch using Gensim.
- **Vector Aggregation**: Represented each review as the average of its word embeddings.

### 3. Model Building
#### Bag of Words and TF-IDF
- Used `GaussianNB` for classification.
- Metrics:
  - **Bag of Words Accuracy**: 57.75%
  - **TF-IDF Accuracy**: 57.875%

#### Word2Vec
- Used `RandomForestClassifier` for classification.
- Metrics:
  - **Word2Vec Accuracy**: 76.75%

### 4. Analysis of Results
- The dataset contains 12,000 rows, which provides a moderate amount of data for training but may still limit the performance of complex models like Word2Vec.
- **BoW and TF-IDF**:
  - Lower accuracy could be due to the inability of these techniques to capture semantic meaning and word context in the reviews.
- **Word2Vec**:
  - Achieved higher accuracy as it leverages distributed word representations, capturing the semantic relationships between words more effectively.
  - However, performance could further improve with more data to enhance embedding quality.

## Tools and Libraries
- **Data Handling**: `pandas`, `numpy`
- **Text Processing**: `re`, `nltk`, `BeautifulSoup`
- **Vectorization**: `sklearn.feature_extraction.text`
- **Modeling**: `sklearn` (GaussianNB, RandomForestClassifier)
- **Word Embeddings**: `gensim`
- **Evaluation**: `sklearn.metrics`

## Future Work
- Experiment with deep learning models like LSTMs or BERT to improve accuracy.
- Fine-tune Word2Vec embeddings using domain-specific text.
- Augment the dataset to include more reviews for better model generalization.

## Acknowledgments
This dataset is taken from Amazon product data by Julian McAuley, UCSD. [Dataset Source](http://jmcauley.ucsd.edu/data/amazon/).

License to the data files belongs to them.
