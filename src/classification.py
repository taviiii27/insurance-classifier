

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def classify_companies(df_companies, taxonomy_labels, threshold=0.15):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df_companies["combined_text"])
    taxonomy_matrix = vectorizer.transform(taxonomy_labels)

    similarity_matrix = cosine_similarity(tfidf_matrix, taxonomy_matrix)

    predicted_labels = []
    for row in similarity_matrix:
        indices = row >= threshold
        labels = [taxonomy_labels[i] for i, flag in enumerate(indices) if flag]
        predicted_labels.append(labels if labels else ["unclassified"])

    df_companies["insurance_label"] = predicted_labels
    return df_companies
