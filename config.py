#Author: Khush Patel, drpatelkhush@gmail.com

#Python imports
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
import seaborn as sns
import pickle

#Sentiment analysis specific imports
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud

#sklearn
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF


#tokens (TO be obtained from twitter developer api, replaced by A to hide identity)

Bearer_token = 'A'
API_key  =  'A'
API_secret_key = 'A'
access_token = 'A'
access_token_secret = 'A'


#Data paths 

#external dataset path for training sentiment analysis model
path = ""

