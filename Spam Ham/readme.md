# Spam vs Ham Classification Project

This project focuses on building machine learning models to classify SMS messages as either `spam` or `ham` (non-spam). The dataset used in this project is the `SMSSpamCollection`, which contains labeled SMS messages.

## Project Workflow

### 1. Data Loading
The dataset is loaded and explored to understand its structure:
- The data consists of two columns: `label` (spam or ham) and `messages` (the SMS content).
- The dataset is loaded using `pandas` and the initial exploration includes checking the shape and previewing the data.
- The dataset contains 5572 rows of labeled SMS messages.

### 2. Data Cleaning and Preprocessing
- **Text Cleaning**: Removing non-alphabetic characters from messages.
- **Case Conversion**: Converting all text to lowercase.
- **Stopword Removal**: Using NLTK's `stopwords`.
- **Lemmatization**: Using NLTK's `WordNetLemmatizer` to normalize words.
- **Corpus Creation**: Preprocessed text messages are stored in a `corpus` for further processing.

### 3. Feature Engineering
#### Bag of Words (BoW)
- Created using `CountVectorizer` with a maximum of 2500 features and bigrams.
- `fit_transform` is used for the training data and `transform` for the test data to prevent data leakage.

#### TF-IDF (Term Frequency-Inverse Document Frequency)
- Implemented using `TfidfVectorizer` with similar parameters as BoW.

#### Word2Vec
- **Word Embeddings**: Trained a `Word2Vec` model from scratch using Gensim.
- **Vector Aggregation**: Represented each message as the average of its word embeddings.

### 4. Model Building
#### Bag of Words and TF-IDF
- Used `Multinomial Naive Bayes` for classification.
- Evaluated using accuracy score and classification report.

#### Word2Vec
- Used `RandomForestClassifier` for classification.
- Evaluated using accuracy score and classification report.

### 5. Results
#### Bag of Words Model:
- Achieved an accuracy of `0.9847533632286996`.
- Detailed classification report included precision, recall, and F1-score for each class.

#### TF-IDF Model:
- Achieved an accuracy of `0.9811659192825112`.
- Similar evaluation metrics as BoW.

#### Word2Vec Model:
- Achieved an accuracy of `0.9704035874439462` using a `RandomForestClassifier`.
- **Reason for Slightly Lower Accuracy**: Word2Vec generates dense, continuous representations of words, which are effective for capturing semantic meaning but may lose certain task-specific nuances captured by BoW or TF-IDF. Additionally, the averaging of word vectors for a document can dilute the contribution of significant words in shorter texts like SMS messages. Furthermore, the dataset has 5572 rows, which, while sufficient for traditional methods like BoW and TF-IDF, may not be large enough to fully leverage the benefits of Word2Vec's embeddings.

## Tools and Libraries
- **Data Handling**: `pandas`, `numpy`
- **Text Processing**: `re`, `nltk`
- **Vectorization**: `sklearn.feature_extraction.text`
- **Modeling**: `sklearn` (Naive Bayes, Random Forest)
- **Word Embeddings**: `gensim`
- **Evaluation**: `sklearn.metrics`
