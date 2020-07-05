import keys
import tweepy
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import os
import re
import pandas as pd
from datetime import date
from BERTModel import Model
import pymongo
import matplotlib.pyplot as plt
import numpy as np

class Sentiment:
    model = Model()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    client = pymongo.MongoClient('172.16.18.46:27017')
    client.test.authenticate('test', 'passw0rd')
    TweetCollection = client.tweets.tweets
    
    def __init__():
        self.totals = []
        for day in TweetCollection.distinct("date"):
            tweets = []
            for tweet in TweetCollection.find({"date": day}):
                tweets.append(tweet["text"])
            predictions = self.model.getPrediction(tweets)
            total = 0
            for prediction in predictions:
                total = prediction[2]/4 + total
            self.totals.append(total/len(tweets))

    def getSentiment(t):
        return self.totals[t]

