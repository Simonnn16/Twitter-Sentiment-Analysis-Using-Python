import pandas as pd
tweets=pd.read_csv('Tweets.csv')
tweets.head()

tweets_df=tweets.drop(tweets[tweets['airline_sentiment_confidence']<0.5].index,axis=0)

tweets_df.shape

X=tweets_df['text']
y=tweets_df['airline_sentiment']

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from nltk.stem import PorterStemmer

stop_words=stopwords.words('english')
stemmer=PorterStemmer()

import re
cleaned_data=[]
for i in range(len(X)):
   tweet=re.sub('[^a-zA-Z]',' ',X.iloc[i])
   tweet=tweet.lower().split()

   tweet=[stemmer.stem(word) for word in tweet if (word not in stop_words)]
   tweet=' '.join(tweet)
   cleaned_data.append(tweet)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000,stop_words=['virginamerica','unit'])
X_fin=cv.fit_transform(cleaned_data).toarray()

sentiment_ordering = ['negative', 'neutral', 'positive']
y = y.apply(lambda x: sentiment_ordering.index(x))

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_fin,y,test_size=0.3)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
from sklearn.metrics import classification_report
cf=classification_report(y_test,y_pred)
print(cf)