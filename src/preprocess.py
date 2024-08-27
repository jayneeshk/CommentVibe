import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load raw data
df = pd.read_csv('data/raw/youtube_comments.csv')

# Data Cleaning
df = df.drop_duplicates()  # Remove duplicate comments
df = df.dropna()  # Drop any rows with missing values

# Text Normalization
df['comment'] = df['comment'].str.lower()  # Convert text to lowercase

# Remove Punctuation and Special Characters
df['comment'] = df['comment'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Remove Stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df['comment'] = df['comment'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Lemmatization
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
df['comment'] = df['comment'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Save the preprocessed data
df.to_csv('data/processed/preprocessed_comments.csv', index=False)

# Vectorization (TF-IDF)
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['comment'])

# Save the vectorized data
pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out()).to_csv('data/processed/vectorized_comments.csv', index=False)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, df['comment'], test_size=0.2, random_state=42)

# Handle Class Imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Save the train-test split data
pd.DataFrame(X_train_sm.toarray(), columns=tfidf.get_feature_names_out()).to_csv('data/processed/X_train.csv', index=False)
pd.DataFrame(X_test.toarray(), columns=tfidf.get_feature_names_out()).to_csv('data/processed/X_test.csv', index=False)
pd.DataFrame(y_train_sm).to_csv('data/processed/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)
