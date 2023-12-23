import pandas as pd
import numpy as np
import chardet
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download NLTK resources
print("Downloading NLTK resources...")
nltk.download("stopwords")
nltk.download("wordnet")


# Clean and tokenize text
def preprocess_text(text):
    # Lowercase, removing special characters and numbers
    text = str(text).lower()
    text = re.sub(
        r"\s+", " ", text
    )  # Convert one or more of any kind of space to single space
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special characters
    text = text.strip()

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in filtered_tokens]


# Display top words for each LDA component
def display_lda_topics(model, feature_names, no_top_words):
    print("# LDA Topics #")
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(
            " ".join(
                [feature_names[i] for i in topic.argsort()[: -no_top_words - 1 : -1]]
            )
        )
    print("\n")


# Display KMeans cluster centers
def display_kmeans_clusters(model, word2vec_model):
    print("# KMeans clusters #")
    for idx, centroid in enumerate(model.cluster_centers_):
        similar_words = word2vec_model.wv.similar_by_vector(centroid, topn=10)
        print(f"Cluster {idx+1}:")
        print(", ".join([word for word, _ in similar_words]))
    print("\n")


# ------------------ Load and process data ------------------
file_name = "BBC Need States - B3_OPEN open ends.csv"
print("Loading data...")
with open(file_name, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
df = pd.read_csv(file_name, encoding=encoding)
print("Cleaning and tokenizing data...")
df_processed = df["B3_OPEN"].apply(preprocess_text)  # Process data

# ------------------ Vectorization ------------------

# Word2vec Vectorization
print("Vectorizing with Word2Vec...")
word2vec_model = Word2Vec(
    df_processed, vector_size=100, window=5, min_count=1, workers=4
)
X_w2v = np.array(
    [
        np.mean(
            [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
            or [np.zeros(word2vec_model.vector_size)],
            axis=0,
        )
        for words in df_processed
    ]
)

# TF-IDF Vectorization
print("Vectorizing with TfidVectorizer...")
tfidf_vectorizer = TfidfVectorizer(
    analyzer="word", tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None
)
X_tfidf = tfidf_vectorizer.fit_transform(df_processed)

# ------------------ LDA and KMeans ------------------

# LatentDirichletAllocation
print("Fitting Latent Dirichlet Allocation model...")
lda = LatentDirichletAllocation(n_components=10)
lda.fit(X_tfidf)
lda_topic_distributions = lda.transform(X_tfidf)
lda_assigned_topics = np.argmax(lda_topic_distributions, axis=1)
df["LDA_Topic"] = lda_assigned_topics

# KMeans
print("Fitting K-Means model...")
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_w2v)
kmeans_clusters = kmeans.predict(X_w2v)
df["KMeans_Cluster"] = kmeans_clusters

# ------------------ Naive Bayes (MultinomialNB and GaussianNB) ------------------

# Process data labels (assume a column called 'Label' in the dataframe)
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(df['Label'])

# GaussianNB Classifier
# print("Training GaussianNB model...")
# X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(X_w2v, y, test_size=0.2)
# gaussian_nb_model = GaussianNB()
# gaussian_nb_model.fit(X_train_w2v, y_train_w2v)
# y_pred = gaussian_nb_model.predict(X_test_w2v)

# MultinomialNB Classifier
# print("Training MultinomialNB model...")
# X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y, test_size=0.2)
# multinomial_nb_model = MultinomialNB()
# multinomial_nb_model.fit(X_train_tfidf, y_train_tfidf)
# y_pred_tfidf = multinomial_nb_model.predict(X_test_tfidf)

# ------------------ Display results ------------------

print("\n### Model categories ###")
display_lda_topics(lda, tfidf_vectorizer.get_feature_names_out(), 10)
display_kmeans_clusters(kmeans, word2vec_model)

# print("# GaussianNB with Word2Vec #\n", classification_report(y_test_w2v, y_pred))
# print("# MultinomialNB with TF-IDF #\n", classification_report(y_test_tfidf, y_pred_tfidf))

# ------------------ Write results to CSV ------------------
print(df.head())
df.to_csv("categorized_responses.csv", index=False)
