# Transformer-Based Sentiment Analysis
import streamlit as st
import pandas as pd
import numpy as np

import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification, TFTrainer, TFTrainingArguments, BertConfig

import SentimentUtils as su

try:
  model = su.load_distilbert_model('./tf_model.h5', './config.json')
except:
  st.error('Model not found')

filename = st.text_input('Enter path to CSV:')
try:
    df, X, y = su.read_data(filename)
    st.write(df.head())
    try:
        batched_data = su.preprocess_data_distilbert(X, y)
        pred_nums, pred_labels = su.predict_distilbert(batched_data)
        df['Sentiment_Prediction'] = pred_labels
        df['Sentiment_Encoded'] = pred_nums
        st.write(df.sample(frac=1).head())
        if st.text_input('Download Predictions? (Y/N)').lower() == 'y':
            savefile = st.text_input('Enter savepath:')
            df.to_csv(savefile)

    except:
      print("Unknown Error preprocessing and predicting data")
except FileNotFoundError:
    st.error('File not found.')

