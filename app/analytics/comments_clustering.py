import time

from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DataCollatorWithPadding

import pandas as pd
import numpy as np
from tqdm import tqdm
import json

from gensim.models import Word2Vec
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
import nltk
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsNERTagger,
    Doc
)

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from scipy.stats import shapiro, levene, ttest_ind
from datetime import datetime, timedelta

import logging
from g4f.client import Client
from ..models_module import work_with_models

NUMBER_OF_CLUSTERS = 5

def comment_vector(tokens, w2v_model):
    """
    Generate a single vector representation of a comment using word embeddings.

    :param tokens: List of tokenized words from a comment.
    :return: Numpy array representing the mean vector of the tokens in the comment.
             Returns a zero vector if no tokens have embeddings in the model.
    """
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if len(vectors) == 0:
        logging.warning(f"list of vectors is empty!")
        return np.zeros(w2v_model.vector_size)
    else:
        logging.info(f"vectors have been collected successfully.")
        return np.mean(vectors, axis=0)


def kmeans_clustering(df, comment_vectors, n_clusters=10, max_iter=300):
    """
    Perform KMeans clustering on comment vectors.

    :param df: DataFrame containing comments and additional metadata.
    :param comment_vectors: Numpy array of comment vector representations.
    :param n_clusters: Number of clusters to form. Default is 10.
    :param max_iter: Maximum number of iterations for the KMeans algorithm. Default is 300.
    :return: Trained KMeans object with cluster assignments.
    """
    scaler = StandardScaler()
    comment_vectors_scaled = scaler.fit_transform(comment_vectors)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=max_iter).fit(comment_vectors_scaled)

    logging.info(f"Done KMeans.")

    return kmeans


class TopicModel:
    """
    A class for topic modeling using BERTopic with a multilingual sentence transformer.

    :param stop_words: list
        A list of stop words to exclude during vectorization.
    """
    def __init__(self, stop_words):
        """
        Initialize the TopicModel with the given stop words and pre-configured models.

        :param stop_words: list
            Stop words to exclude during topic modeling.
        """
        self.stop_words = stop_words
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words=list(self.stop_words)),
            nr_topics=15,
            verbose=True
        )

    def forward(self, comments):
        """
        Fit and transform the comments into topics and probabilities.

        :param comments: list
            List of text comments.
        :return: tuple
            Topics and probabilities for each comment.
        """
        return self.topic_model.fit_transform(comments)


def get_bertopic_clusters(df, model: TopicModel):
    """
    Assign BERTopic clusters to a DataFrame and filter out insignificant clusters.

    :param df: pandas.DataFrame
        DataFrame containing a column 'textDisplay' with comments.
    :param model: TopicModel
        An instance of the TopicModel class.
    :return: pandas.DataFrame
        DataFrame with assigned cluster labels, filtered by significance.
    """
    comments = df['textDisplay'].fillna('').values.tolist()
    topics, probs = model.forward(comments)
    topic_model = model.topic_model
    topic_s = topic_model.get_topics()
    indeces = [k for k, v in topic_s.items()]

    df['cluster'] = topics
    df['topic_raw'] = topic_model.get_topic_info().Name

    good_clusters = []
    for i in range(len(indeces)):
        if 0.005 * len(df) < topic_model.get_topic_info().iloc[i].Count < 0.4 * len(df) and topic_model.get_topic_info().iloc[i].Count > 10: # and topic_coherence_scores[i] > 0.3
            good_clusters.append(indeces[i])

    logging.info(f"Collected good clusters for BERTopic")

    return df.isin({'cluster': good_clusters})


def get_kmeans_clusters(df):
    """
    Apply KMeans clustering to comments and filter out insignificant clusters.

    :param df: pandas.DataFrame
        DataFrame containing a column 'comment_vector' with precomputed vectors.
    :return: pandas.DataFrame
        DataFrame with filtered cluster labels added as 'cluster_km'.
    """
    comment_vectors = np.vstack(df['comment_vector'].values)
    kmeans = kmeans_clustering(df, comment_vectors, n_clusters=15, max_iter=300)

    clusters = kmeans.labels_
    df['cluster_km'] = clusters

    # print(len(clusters), clusters)

    good_clusters = []
    for i in range(len(df['cluster_km'].unique())):
        if 0.005 * len(df) < dict(df['cluster_km'].value_counts())[i] < 0.4 * len(df) and dict(df['cluster_km'].value_counts())[i] > 10: # and topic_coherence_scores[i] > 0.3
            good_clusters.append(i)

    logging.info(f"Collected good clusters for KMeans")

    return df[df['cluster_km'].isin(good_clusters)]


def title_clusters(df_bert, df_kmeans):
    """
    Generate titles for clusters using GPT model for both BERTopic and KMeans clusters.

    :param df_bert: pandas.DataFrame
        DataFrame with BERTopic clusters labeled.
    :param df_kmeans: pandas.DataFrame
        DataFrame with KMeans clusters labeled.
    :return: dict
        Dictionary mapping cluster titles to example comments.
    """
    answer = {}
    client = Client()

    for cluster_id in tqdm(df_kmeans['cluster_km'].unique()):
        km_i = df_kmeans[df_kmeans['cluster_km'] == cluster_id]['textDisplay'].head(10).to_list()

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Не отвечай ничего кроме названия кластера"},
                {"role": "user", "content": f"Дай название кластеру: {km_i}"}
            ],
        )
        title = (response.choices[0].message.content)
        time.sleep(8)
        answer[title] = (km_i, len(df_kmeans[df_kmeans['cluster_km'] == cluster_id]['textDisplay'].to_list()))

    logging.info(f"KMeans clusters have been titled successfully.")

    for cluster_id in tqdm(df_bert['cluster'].unique()):
        bert_i = df_bert[df_bert['cluster'] == cluster_id]['textDisplay'].head(10).to_list()

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Не отвечай ничего кроме названия кластера"},
                {"role": "user", "content": f"Дай название кластеру: {bert_i}"}
            ],
        )
        title = (response.choices[0].message.content)
        time.sleep(8)

        answer[title] = (bert_i, len(df_bert[df_bert['cluster'] == cluster_id]['textDisplay'].to_list()))

    logging.info(f"BERTopic clusters have been titled successfully.")

    if (len(answer) > 10):
        sorted_answer = dict(sorted(answer.items(), key=lambda item: item[1][1]))
        answer = dict(list(sorted_answer.items())[:NUMBER_OF_CLUSTERS])

    return answer


def clustering(video_id: str):
    df = work_with_models.get_comments_df(video_id)
    nltk.download('punkt_tab')
    df['tokens'] = df['textDisplay'].astype(str).apply(word_tokenize)
    w2v_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=2, sg=1)
    df['comment_vector'] = df['tokens'].apply(lambda tokens: comment_vector(tokens, w2v_model))

    stop_words = np.loadtxt('analytics/stopwords-ru.txt', dtype=str, usecols=0)
    model = TopicModel(stop_words)
    df_bert = get_bertopic_clusters(df, model)
    df_kmeans = get_kmeans_clusters(df)
    titles = pd.DataFrame(title_clusters(df_bert, df_kmeans))
    # result = titles.iloc[:, : 5]
    print(titles.to_string())
    # return result
