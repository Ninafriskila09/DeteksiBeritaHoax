import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import joblib

# Load data
data = pd.read_excel('dataset_clean.xlsx')

# Preprocessing data
X_raw = data["clean_text"]
y_raw = data["Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
vectorizer.fit(X_train)
X_train_TFIDF = vectorizer.transform(X_train)
X_test_TFIDF = vectorizer.transform(X_test)

chi2_features = SelectKBest(chi2, k=500)
X_kbest_features = chi2_features.fit_transform(X_train_TFIDF, y_train)

# Train model
NB = GaussianNB()
X_train_dense = csr_matrix.toarray(X_kbest_features)
NB.fit(X_train_dense, y_train)

# Save model and preprocessing tools
joblib.dump(NB, 'naive_bayes_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(chi2_features, 'chi2_features.joblib')
