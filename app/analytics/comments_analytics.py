import logging

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
import psycopg2 as pg
from ..models_module import work_with_models
from ..models_module import db_sessions

# from gensim.models import Word2Vec
# from nltk.tokenize import word_tokenize
# import nltk
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


def process_sentiment_analysis(texts, tokenizer, model, batch_size=16, device='cpu') -> np.array:
    """
    Perform sentiment analysis on a list of text inputs using a given model.

    :param texts: List of strings to analyze.
    :param model: Pretrained model for sentiment analysis (e.g., Hugging Face transformer).
    :param batch_size: Number of texts to process in a single batch. Default is 16.
    :param device: Device to use for model inference ('cpu' or 'cuda'). Default is 'cpu'.
    :return: Numpy array of predicted sentiment classes for each text (e.g., 0: neutral, 1: positive, 2: negative).
    """
    sentiments = []
    model = model.to(device)

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            try:
                inputs = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            except:
                print(batch)
                break
            outputs = model(**inputs)
            predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted = torch.argmax(predicted, dim=1).cpu().numpy()
            sentiments.append(predicted)

    sentiments = np.concatenate(sentiments)

    return sentiments


def sentiment_analytics(sentiments, timestamp=''):
    """
    Calculate sentiment statistics for a given array of sentiment predictions.

    :param sentiments: Numpy array of sentiment predictions (0: neutral, 1: positive, 2: negative).
    :param timestamp: Optional timestamp for the analysis. Default is an empty string.
    :return: Dictionary containing sentiment counts, percentages, a proportion metric, and the timestamp.
    """
    answer = {}

    mask_positive = sentiments == 1
    mask_negative = sentiments == 2
    mask_neutral = sentiments == 0

    cnt_positive = np.sum(mask_positive)
    cnt_negative = np.sum(mask_negative)
    cnt_neutral = np.sum(mask_neutral)

    cnt_comments = cnt_positive + cnt_negative + cnt_neutral

    answer['positive'] = cnt_positive
    answer['negative'] = cnt_negative
    answer['neutral'] = cnt_neutral
    answer['positive_perc'] = (100 * cnt_positive / cnt_comments)
    answer['negative_perc'] = (100 * cnt_negative / cnt_comments)
    answer['neutral_perc'] = (100 * cnt_neutral / cnt_comments)
    answer['proportion_metric'] = np.mean(sentiments)
    answer['timestamp'] = timestamp

    return answer


def plot_dynamics(values, title="Динамика значений",
                  xlabel="Дата и время", ylabel="Значение"):
    """
    Plot sentiment dynamics over time.

    :param values: List of dictionaries containing timestamps and sentiment percentages
                   (keys: 'timestamp', 'positive_perc', 'negative_perc', 'neutral_perc').
    :param title: Title of the plot. Default is "Динамика значений".
    :param xlabel: Label for the X-axis. Default is "Дата и время".
    :param ylabel: Label for the Y-axis. Default is "Значение".
    :return: None. Displays the plot.
    """

    dates = [stat['timestamp'] for stat in values]
    values1 = [stat['positive_perc'] for stat in values]
    values2 = [stat['negative_perc'] for stat in values]
    values3 = [stat['neutral_perc'] for stat in values]
    label1 = 'Процент позитивных комментариев'
    label2 = 'Процент негативных комментариев'
    label3 = 'Процент нейтральных комментариев'

    if not values1 or not values2:
        raise ValueError("Оба списка значений не должны быть пустыми.")

    plt.figure(figsize=(10, 6))
    plt.plot(dates, values1, marker='o', linestyle='-', color='b', label=label1)
    plt.plot(dates, values2, marker='s', linestyle='--', color='r', label=label2)
    # plt.plot(dates, values3, marker='s', linestyle='-.', color='grey', alpha=0.5, label=label3)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate(rotation=60, ha='right')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


def normal_dist_test(sample: np.array) -> bool:
     """
     Test whether a given sample follows a normal distribution.

     :param sample: Numpy array of sample values.
     :return: True if the sample follows a normal distribution, False otherwise.
     """
     mean_sample = []
     for i in range(500):
         subsample = np.random.choice(sample, size=100)
         mean_sample.append(np.mean(subsample))

     mean_sample = np.array(mean_sample)
     result = shapiro(mean_sample).statistic

     if result > 0.05:
         return True
     return False


def var_equality_test(sample1: np.array, sample2: np.array) -> bool:
    """
    Test whether two samples have equal variance.

    :param sample1: First numpy array of sample values.
    :param sample2: Second numpy array of sample values.
    :return: True if the variances are equal, False otherwise.
    """
    result = levene(sample1, sample2).statistic

    if result > 0.05:
        return True
    return False


def are_emotions_same(s1: np.array, s2: np.array) -> bool:
    """
    Determine if two sentiment samples are statistically similar using hypothesis testing.

    :param s1: First numpy array of sentiment predictions.
    :param s2: Second numpy array of sentiment predictions.
    :return: True if the sentiment distributions are statistically similar, False otherwise.
    """
    sz = np.min([len(s1), len(s2)])
    sample1 = np.random.choice(s1, size=sz)
    sample2 = np.random.choice(s2, size=sz)

    if not normal_dist_test(sample1) or not normal_dist_test(sample2):
        print('not normal distribution')
        return False

    result = ttest_ind(sample1, sample2, equal_var=var_equality_test(sample1, sample2)).pvalue

    print(f't-test pvalue: {result}')

    if result > 0.05:
        return True
    return False


def get_emotional_dynamics_in_video(video_id, df, start, end, tokenizer, model, device, detailing='h'):
    """
    Analyze the emotional dynamics of comments for a specific video within a time range.

    :param video_id: ID of the video to analyze.
    :param df: DataFrame containing comments data with 'video_id' and 'publishedAt' columns.
    :param start: Start time (datetime object) for the analysis.
    :param end: End time (datetime object) for the analysis.
    :param model: Pretrained model for sentiment analysis.
    :param device: Device for model inference ('cpu' or 'cuda').
    :param detailing: Granularity of the analysis ('m': minute, 'h': hour, 'd': day). Default is 'h'.
    :return: List of dictionaries containing sentiment analytics for each time interval.
    """
    sample = df[df['videoId'] == video_id]
    max_iter = 100
    comments = np.array([])
    answer = []

    iter = 0

    current_time = start
    while current_time <= end:
        prev_current_time = current_time
        if detailing == 'm':
            addition = sample[(sample.publishedAt >= current_time) & (sample.publishedAt < current_time + timedelta(minutes=1))]
            current_time += timedelta(minutes=1)
        elif detailing == 'h':
            addition = sample[(sample.publishedAt >= current_time) & (sample.publishedAt < current_time + timedelta(hours=1))]
            current_time += timedelta(hours=1)
        elif detailing == 'd':
            addition = sample[(sample.publishedAt >= current_time) & (sample.publishedAt < current_time + timedelta(days=1))]
            current_time += timedelta(days=1)

        if addition.empty:
            continue

        iter += 1

        result = process_sentiment_analysis(addition['textDisplay'].fillna('').values.tolist(), tokenizer, model, device=device)
        if (len(comments) == 0):
            comments = np.array(result)
        else:
            comments = np.concatenate((comments, result))

        analytics = sentiment_analytics(comments, prev_current_time)
        answer.append(analytics)

        if iter >= max_iter:
            print('Limit reached')
            break

    return answer


def comments_emotional_analytics(video_id: str):
    start_time = work_with_models.video_published_time(video_id)
    date_object = datetime.strptime(start_time, "%Y-%m-%d")

    finish_date = date_object + timedelta(days=7)
    finish_time = finish_date.strftime("%Y-%m-%d")
    df = work_with_models.get_comments_df(video_id)

    tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment')
    model = AutoModelForSequenceClassification.from_pretrained(
        'blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True)
    df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.tz_localize('UTC')
    print(df.head(3))

    texts = df['textDisplay'].fillna('').values.tolist()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # sentiments = process_sentiment_analysis(texts, model, batch_size=16, device=device)
    dyn = get_emotional_dynamics_in_video(video_id, df, pd.Timestamp(start_time, tz='UTC'),
                                          pd.Timestamp(finish_time, tz='UTC'), tokenizer, model,
                                          device, 'h')
    print(dyn)
    # plot_dynamics(dyn)


# BERTopic
#
# stop_words = np.loadtxt('stopwords-ru.txt', dtype=str, usecols=0)
# comments = df['textDisplay'].fillna('').values.tolist()
# sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#
# topic_model = BERTopic(
#     embedding_model=sentence_model,
#     vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words=list(stop_words)),
#     nr_topics=15,
#     verbose=True
# )
#
# # Кластеризация текстов
# topics, probs = topic_model.fit_transform(comments)
#
# # Результаты кластеризации
# # print("Темы для каждого комментария:")
# # for comment, topic in zip(comments, topics):
# #     print(f"Комментарий: {comment}\nТема: {topic}\n")
#
# # Визуализация тем
# # topic_model.visualize_topics()