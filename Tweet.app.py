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


def app():

    st.title(
        "PH 1977 Dr. Yaseen SPH  HW 3: Tweet sentiment analysis and topic modeling"
    )

    activities = ["Tweet Sentiment Analysis", "Topic Modeling"]

    choice = st.sidebar.selectbox("Select Your Activity", activities)

    if choice == "Tweet Sentiment Analysis":

        st.subheader("Analyze the tweets from any twitter handle")

        st.subheader("This tool performs the following tasks :")

        st.write("1. Fetches the 10 most recent tweets from the given twitter handle")
        st.write("2. Generates a Word Cloud")
        st.write(
            "3. Performs Sentiment Analysis a displays it in form of a Bar Graph using TextBlob"
        )
        st.write(
            "4. Performs Sentiment Analysis a displays it in form of a Bar Graph using IMDB Logistic Regression Method"
        )

        raw_text = st.text_area("Enter the exact twitter handle to study (without @)")

        Analyzer_choice = st.selectbox(
            "Select the Activities",
            [
                "Show Recent Tweets",
                "Generate WordCloud",
                "Visualize the Sentiment Analysis (TextBlob)",
                "Visualize the Sentiment Analysis (IMDB Logistic Regression Method)",
            ],
        )

        if st.button("Analyze"):

            if Analyzer_choice == "Show Recent Tweets":

                st.success("Fetching last 10 Tweets")

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

    if choice == "Visualize the Sentiment Analysis (TextBlob)":

        st.subheader(
            "This tool fetches the last 100 tweets from the twitter handel & Performs the following tasks"
        )

        st.write("1. Converts it into a DataFrame")
        st.write("2. Cleans the text")
        st.write(
            "3. Analyzes Subjectivity of tweets and adds an additional column for it using TextBlob"
        )
        st.write(
            "4. Analyzes Polarity of tweets and adds an additional column for it using TextBlob"
        )
        st.write(
            "5. Analyzes Sentiments of tweets and adds an additional column for it using TextBlob"
        )

        user_name = st.text_area(
            "*Enter the exact twitter handle of the Personality (without @)*"
        )

        def get_data(user_name):

            posts = api.user_timeline(
                screen_name=user_name, count=200, lang="en", tweet_mode="extended"
            )

            df = pd.DataFrame([tweet.full_text for tweet in posts], columns=["Tweets"])

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

    else:
        if choice == "Visualize the Sentiment Analysis (Logistic Regression)":

            st.subheader(
                "This tool fetches the last 200 tweets from the twitter handel & Performs the following tasks"
            )

            st.write("1. Converts it into a DataFrame")
            st.write("2. Cleans the text")
            st.write(
                "3. Analyzes Subjectivity of tweets and adds an additional column for it using TextBlob"
            )
            st.write(
                "4. Analyzes Polarity of tweets and adds an additional column for it using TextBlob"
            )
            st.write(
                "5. Analyzes Sentiments of tweets and adds an additional column for it using TextBlob"
            )

            user_name = st.text_area(
                "*Enter the exact twitter handle of the Personality (without @)*"
            )

            def get_data(user_name):

                posts = api.user_timeline(
                    screen_name=user_name, count=200, lang="en", tweet_mode="extended"
                )

                df = pd.DataFrame(
                    [tweet.full_text for tweet in posts], columns=["Tweets"]
                )

                def cleanTxt(text):
                    text = re.sub("@[A-Za-z0–9]+", "", text)  # Removing @mentions
                    text = re.sub("#", "", text)  # Removing '#' hash tag
                    text = re.sub("RT[\s]+", "", text)  # Removing RT
                    text = re.sub("https?:\/\/\S+", "", text)  # Removing hyperlink
                    return text

                # Clean the tweets
                df["Tweets"] = df["Tweets"].apply(cleanTxt)

                logistic_regression = pickle.load(open("logistic.pkl", "rb"))

                cv = TfidfVectorizer(min_df=5, ngram_range=(1, 3))

                dtm = cv.fit_transform(df.Tweets)

                df["sentiment_score_imdb"] = logistic_regression.predict(dtm)

                df["sentiment_score_imdb"] = df["sentiment_score_imdb"].map(
                    {1: "Positive", 0: "Negative"}
                )

                return df

            # if st.button("Topic Modeling"):

            #     st.success("Fetching Last 200 Tweets")

            #     df = get_data(user_name)

            #     st.write(df)

    st.subheader("Khush Patel, Khush.A.Patel@uth.tmc.edu")


if __name__ == "__main__":
    app()

