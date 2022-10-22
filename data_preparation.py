#Author: Khush Patel, drpatelkhush@gmail.com

from config import *

#Authenticate tokens

# Authenticate object
authenticate = tweepy.OAuthHandler(API_key, API_secret_key) 
    
# Setting the access token and access token secret
authenticate.set_access_token(access_token, access_token_secret) 
    
# Creating the API object while passing in auth information
api = tweepy.API(authenticate, wait_on_rate_limit = True)

# Extracting 500 tweets from the twitter user @UTexasSPH for trial, number can be increased later. 
posts = api.user_timeline(screen_name='UTexasSPH', count = 500, lang ="en", tweet_mode="extended")
#Eg output: 
"""1) RT @texas_cares: ¿Estaría Usted agradecido por información?
Obtenga su Covid-19 test de anticuerpos conveniente y gratuito esta semana en u…"""

# Cleaning text:
#Creating a dataframe: 
df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])

# Create a function to clean the tweets
def cleanTxt(text):
  text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
  text = re.sub('#', '', text) # Removing '#' hash tag
  text = re.sub('RT[\s]+', '', text) # Removing RT
  text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
  return text

# Cleaning the tweets
df['Tweets'] = df['Tweets'].apply(cleanTxt)

# Removing NaN VALUES AND EMPTY STRINGS:
df.dropna(inplace=True)

blanks = []  

for i,tw in df.itertuples():
    if type(tw)==str:            
        if tw.isspace():         
            blanks.append(i)     

df.drop(blanks, inplace=True)
df.to_csv("clean_texts.csv", index=False)