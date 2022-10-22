from config import *


df = pd.read_csv("sentiment_analysis.csv")  #generated from sentiment analysis file
#First using LDA
LDA = LatentDirichletAllocation(n_components=4,random_state=42)
cv= TfidfVectorizer(min_df=5, ngram_range=(1,3))
dtm = cv.fit_transform(df.Tweets)

print(cv.get_feature_names()[:10])  #CV features

LDA.fit(dtm)
"""#LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                          evaluate_every=-1, learning_decay=0.7,
                          learning_method='batch', learning_offset=10.0,
                          max_doc_update_iter=100, max_iter=10,
                          mean_change_tol=0.001, n_components=4, n_jobs=None,
                          perp_tol=0.1, random_state=42, topic_word_prior=None,
                          total_samples=1000000.0, verbose=0)"""
                          

topic_results = LDA.transform(dtm)
df['Topic_LDA'] = topic_results.argmax(axis=1)

#Printing top words for each category identified by topic modeling
for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index} using LDA')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')
    
    
#Topic modeling using NMF

nmf_model = NMF(n_components=4,random_state=42)
nmf_model.fit(dtm)

topic_results = nmf_model.transform(dtm)
df['Topic_NMF'] = topic_results.argmax(axis=1)


df.to_csv("sentiment_analysis_topic_modeling.csv", index=False)



#Printing top words for each category identified by topic modeling
for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index} using NMF')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')