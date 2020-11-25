#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Python imports
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
import seaborn as sns
import pickle
import streamlit as st
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Sentiment analysis specific imports
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud

# sklearn
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF


# In[2]:


Bearer_token = "AAAAAAAAAAAAAAAAAAAAAIw7KAEAAAAA8sB30bk6YmLoZ44O4SYsnVm77Xo%3Du7pjIJXUUR9yCgXFa68CGtT89DJHnRf06RHfUPJTmcdBiSzluZ"

API_key = "6YwLm5RrFM2svJwZo2CSXCpbS"
API_secret_key = "bKvPXAG8tHy0qEzazgEX8rqqrcqspKWfqpJKn03ESBOBuxhgFg"

access_token = "1027516755198009345-alcKwxRGMibuhplbS1wx47fnmaUkik"
access_token_secret = "avl6MD9anfLIKgALs14JEohXN7NfuDN6cmm9YRvLpT593"

# Create the authentication object
authenticate = tweepy.OAuthHandler(API_key, API_secret_key)

# Set the access token and access token secret
authenticate.set_access_token(access_token, access_token_secret)

# Creating the API object while passing in auth information
api = tweepy.API(authenticate, wait_on_rate_limit=True)


# In[ ]:


def app():

    import warnings

    warnings.filterwarnings("ignore")

    st.set_option("deprecation.showPyplotGlobalUse", False)

    st.title(
        "PH 1977 Dr. Yaseen SPH  HW 3: Tweet sentiment analysis and topic modeling"
    )

    activities = ["Tweet Sentiment Analysis", "Topic Modeling"]

    choice = st.sidebar.selectbox("Select Your Activity", activities)

    if choice == "Tweet Sentiment Analysis":

        st.subheader("You have selected Sentiment Analysis of Tweet")

        st.subheader("Analyze the tweets from any twitter handle")

        st.subheader("Based on our class code from module 7, following will be done ")

        st.write("1. Fetches the 5 most recent tweets from the given twitter handle")
        st.write("2. Generates a Word Cloud")
        st.write(
            "3. Performs Sentiment Analysis using TextBlob a displays it in form of a Bar Graph"
        )
        
        raw_text = st.text_area("Enter the exact twitter handle to study (without @)")

        Analyzer_choice = st.selectbox(
            "Select the Activities",
            [
                "Show Recent Tweets",
                "Generate WordCloud",
                "Visualize the Sentiment Analysis (TextBlob)"
            ],
        )

        if st.button("Analyze"):

            if Analyzer_choice == "Show Recent Tweets":

                st.success("Fetching last 5 Tweets")

                def Show_Recent_Tweets(raw_text):

                    # Extract 200 tweets from the twitter user
                    posts = api.user_timeline(
                        screen_name=raw_text,
                        count=200,
                        lang="en",
                        tweet_mode="extended",
                    )

                    def get_tweets():

                        l = []
                        i = 1
                        for tweet in posts[:5]:
                            l.append(tweet.full_text)
                            i = i + 1
                        return l

                    recent_tweets = get_tweets()
                    return recent_tweets

                recent_tweets = Show_Recent_Tweets(raw_text)

                st.write(recent_tweets)

            elif Analyzer_choice == "Generate WordCloud":

                st.success("Generating Word Cloud")

                def gen_wordcloud():

                    posts = api.user_timeline(
                        screen_name=raw_text,
                        count=200,
                        lang="en",
                        tweet_mode="extended",
                    )

                    # Create a dataframe with a column called Tweets
                    df = pd.DataFrame(
                        [tweet.full_text for tweet in posts], columns=["Tweets"]
                    )
                    # word cloud visualization
                    allWords = " ".join([twts for twts in df["Tweets"]])
                    wordCloud = WordCloud(
                        width=500, height=300, random_state=21, max_font_size=110
                    ).generate(allWords)
                    plt.imshow(wordCloud, interpolation="bilinear")
                    plt.axis("off")
                    plt.savefig("WC.jpg")
                    img = Image.open("WC.jpg")
                    return img

                img = gen_wordcloud()

                st.image(img)

            else:

                def Plot_Analysis():

                    st.success(
                        "Generating Visualisation for Sentiment Analysis using TextBlob"
                    )

                    posts = api.user_timeline(
                        screen_name=raw_text,
                        count=200,
                        lang="en",
                        tweet_mode="extended",
                    )

                    df = pd.DataFrame(
                        [tweet.full_text for tweet in posts], columns=["Tweets"]
                    )

                    # Create a function to clean the tweets
                    def cleanTxt(text):
                        text = re.sub("@[A-Za-z0–9]+", "", text)  # Removing @mentions
                        text = re.sub("#", "", text)  # Removing '#' hash tag
                        text = re.sub("RT[\s]+", "", text)  # Removing RT
                        text = re.sub("https?:\/\/\S+", "", text)  # Removing hyperlink

                        return text

                    # Clean the tweets
                    df["Tweets"] = df["Tweets"].apply(cleanTxt)

                    def getSubjectivity(text):
                        return TextBlob(text).sentiment.subjectivity

                    # Create a function to get the polarity
                    def getPolarity(text):
                        return TextBlob(text).sentiment.polarity

                    # Create two new columns 'Subjectivity' & 'Polarity'
                    df["Subjectivity"] = df["Tweets"].apply(getSubjectivity)
                    df["Polarity"] = df["Tweets"].apply(getPolarity)

                    def getAnalysis(score):
                        if score < 0:
                            return "Negative"
                        elif score == 0:
                            return "Neutral"
                        else:
                            return "Positive"

                    df["Analysis"] = df["Polarity"].apply(getAnalysis)

                    return df

                df = Plot_Analysis()

                st.write(sns.countplot(x=df["Analysis"], data=df))

                st.pyplot(use_container_width=True)

            

    else:

        st.subheader("You have topic modeling of tweets")

        st.subheader("Classify tweets into topics")

        st.subheader("Based on our class code from module 7, following will be done ")

        st.write("1. Classify tweet into topics")
        st.write(
            "2. Most common 5 words from each topic will be displayed making it easier to guess the topic label"
        )

        raw_text = st.text_area("Enter the exact twitter handle to study (without @)")

        Analyzer_choice = st.selectbox("Select the Method", ["LDA", "NMF"],)

        if st.button("Analyze"):

            if Analyzer_choice == "LDA":

                st.success("Performing topic modeling using LDA")

                cv = TfidfVectorizer(min_df=5, ngram_range=(1, 3))

                def Plot_Analysis():

                    st.success("Generating Topic Modeling using LDA")

                    posts = api.user_timeline(
                        screen_name=raw_text,
                        count=200,
                        lang="en",
                        tweet_mode="extended",
                    )

                    df = pd.DataFrame(
                        [tweet.full_text for tweet in posts], columns=["Tweets"]
                    )

                    # Create a function to clean the tweets
                    def cleanTxt(text):
                        text = re.sub("@[A-Za-z0–9]+", "", text)  # Removing @mentions
                        text = re.sub("#", "", text)  # Removing '#' hash tag
                        text = re.sub("RT[\s]+", "", text)  # Removing RT
                        text = re.sub("https?:\/\/\S+", "", text)  # Removing hyperlink

                        return text

                    # Clean the tweets
                    df["Tweets"] = df["Tweets"].apply(cleanTxt)

                    LDA = LatentDirichletAllocation(n_components=5, random_state=42)

                    dtm = cv.fit_transform(df.Tweets)

                    topic_results = LDA.fit_transform(dtm)

                    df["Topic_LDA"] = topic_results.argmax(axis=1)

                    return df

                df = Plot_Analysis()

                st.write("Showing last 10 recent tweets and topics")

                st.write(df.head(10))

                LDA = LatentDirichletAllocation(n_components=5, random_state=42)

                dtm = cv.fit_transform(df.Tweets)

                topic_results = LDA.fit_transform(dtm)

                Top_5_words = []

                for index, topic in enumerate(LDA.components_):
                    a = [cv.get_feature_names()[i] for i in topic.argsort()[-5:]]
                    Top_5_words.append({index: a})

                st.write("Top 5 words for each topic")

                st.write(Top_5_words)

            else:

                st.success("Performing topic modeling using NMF")

                cv = TfidfVectorizer(min_df=5, ngram_range=(1, 3))

                def Plot_Analysis():

                    st.success("Generating Topic Modeling using NMF")

                    posts = api.user_timeline(
                        screen_name=raw_text,
                        count=200,
                        lang="en",
                        tweet_mode="extended",
                    )

                    df = pd.DataFrame(
                        [tweet.full_text for tweet in posts], columns=["Tweets"]
                    )

                    # Create a function to clean the tweets
                    def cleanTxt(text):
                        text = re.sub("@[A-Za-z0–9]+", "", text)  # Removing @mentions
                        text = re.sub("#", "", text)  # Removing '#' hash tag
                        text = re.sub("RT[\s]+", "", text)  # Removing RT
                        text = re.sub("https?:\/\/\S+", "", text)  # Removing hyperlink

                        return text

                    # Clean the tweets
                    df["Tweets"] = df["Tweets"].apply(cleanTxt)

                    return df

                df = Plot_Analysis()

                nmf_model = NMF(n_components=5, random_state=42)

                dtm = cv.fit_transform(df.Tweets)

                topic_results = nmf_model.fit_transform(dtm)

                df["Topic_nmf"] = topic_results.argmax(axis=1)

                st.write("Showing last 10 recent tweets and topics")

                st.write(df.head(10))

                st.write("Topics Distribution")

                nmf_model = NMF(n_components=5, random_state=42)

                dtm = cv.fit_transform(df.Tweets)

                topic_results = nmf_model.fit_transform(dtm)

                Top_5_words = []

                for index, topic in enumerate(nmf_model.components_):
                    a = [cv.get_feature_names()[i] for i in topic.argsort()[-5:]]
                    Top_5_words.append({index: a})

                st.write(
                    "Top 5 words for each topic: It can be used to understand topics clearly"
                )

                st.write(Top_5_words)

    st.subheader("Khush Patel, Khush.A.Patel@uth.tmc.edu")


if __name__ == "__main__":
    app()


