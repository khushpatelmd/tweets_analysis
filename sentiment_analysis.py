from config import * 

df = pd.read_csv("clean_texts.csv")

#Sentiment analysis

#Textblob approach
# Creating a function to get the subjectivity
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

# Creating a function to get the polarity
def getPolarity(text):
  return  TextBlob(text).sentiment.polarity


# Creating two new columns 'Subjectivity' & 'Polarity'
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)

# Creating a function to compute negative (-1), neutral (0) and positive (+1) analysis
def getAnalysis(score):
  if score < 0:
    return 'Negative'
  elif score == 0:
    return 'Neutral'
  else:
    return 'Positive'
    
df['Analysis_textblob'] = df['Polarity'].apply(getAnalysis)

# Plotting and visualizing the counts
plt.title('Sentiment Analysis using textblob')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis_textblob'].value_counts().plot(kind = 'bar')
plt.savefig("Sentiment_analysis_text_blob.svg")


#We can train a new model on existing labelled datasets instead of using textblob. To do so, lets use IMDB sentiment analysis dataset due to ease of availability but the approach can be extended to any labelled datasets. 

#Imdb dataset model
train_path  = 'path/train'
test_path = 'path/test'

reviews_train = load_files(train_path)
text_train, y_train = reviews_train.data, reviews_train.target
print("length of text_train: {}".format(len(text_train)))
print("text_train[6]:\n{}".format(text_train[6]))

reviews_test = load_files(test_path)
text_test, y_test = reviews_test.data, reviews_test.target
print("Number of documents in test data: {}".format(len(text_test)))
print("Samples per class (test): {}".format(np.bincount(y_test)))
text_test  = [doc.replace(b"<br />", b" ") for doc in text_test]

#remove HTML line breaks (<br />)
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

#Using Tfidfvectorizer and then using Logistic Regression 

cv= TfidfVectorizer(min_df=5, ngram_range=(1,3))
text_train = cv.fit_transform(text_train)
model_logistic = LogisticRegression.fit(text_train, y_train)


cv = pickle.load(open("tfidf_logistic.pkl", "rb"))
dtm = cv.transform(df.Tweets)
model_logistic = pickle.load(open("logistic.pkl", "rb"))

#Predicting sentiment using Logistic Regression model

df['sentiment_score_imdb'] = model_logistic.predict(dtm)

df['sentiment_score_imdb'] = df['sentiment_score_imdb'].map({1:'Positive', 0:'Negative'})

#Plotting sentiment analysis using Logistic Regression
plt.title('Sentiment Analysis using IMDB classifier')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['sentiment_score_imdb'].value_counts().plot(kind = 'bar')
plt.savefig("Sentiment_analysis_transfer_learning.svg")


df.to_csv("sentiment_analysis.csv", index=False)