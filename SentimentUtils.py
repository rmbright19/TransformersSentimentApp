# Transformer-Based Sentiment Analysis Utils
import pandas as pd
import numpy as np
import tensorflow as tf

from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification, TFTrainer, TFTrainingArguments, BertConfig

def load_distilbert_model(model_path='./tf_model.h5', config_path='./config.json'):
    from transformers import DistilBertConfig
    from transformers import TFDistilBertForSequenceClassification

    config = DistilBertConfig.from_json_file(config_path)
    model_reloaded = TFDistilBertForSequenceClassification.from_pretrained(model_path, config=config)
    return model_reloaded

def read_data(csv_path, tweet_col='text', label_col=None, shuffle=True):
    if label_col:
        df = pd.read_csv(csv_path, usecols=[tweet_col, label_col])
        if shuffle:
            df = df.sample(frac=1)
        X = df['text'].to_list()
        y = df[label_col].to_list()
    else:
        df = pd.read_csv(csv_path, usecols=[tweet_col])
        if shuffle:
            df = df.sample(frac=1)
        X = df['text'].to_list()
        y=None

    return df, X, y

def preprocess_data_distilbert(X, y=None, orig_checkpoint='distilbert-base-uncased'):
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained(orig_checkpoint, num_labels=3)
    encodings = tokenizer(X, truncation=True, padding=True)
    print(len(encodings))
    if not y:
        y = np.zeros(len(X))
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), y))
    dataset_batched = dataset.batch(16)

    return dataset_batched

