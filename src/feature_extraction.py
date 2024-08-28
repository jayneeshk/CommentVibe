import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the preprocessed comments data
df = pd.read_csv('data/processed/preprocessed_comments.csv')

# Handle NaN values by replacing them with an empty string
df['comment'] = df['comment'].fillna('')

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,  # Limit to top 5000 features
    stop_words='english',  # Removes stopwords during vectorization
    ngram_range=(1, 2)  # Considers unigrams and bigrams
)

# Fit and transform the comment text to create TF-IDF features
X = tfidf.fit_transform(df['comment'])

# Save the TF-IDF matrix (features) to a file
pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out()).to_csv('data/processed/tfidf_features.csv', index=False)

# Save the TF-IDF vectorizer for future use
joblib.dump(tfidf, 'data/models/tfidf_vectorizer.pkl')
