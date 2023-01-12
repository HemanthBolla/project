"""from flask import Flask , render_template, url_for,request
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score,plot_confusion_matrix,classification_report ,confusion_matrix
from wordcloud import WordCloud
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

def spacy_tokenizer(sentence):
    mytokens=parser(sentence)
    mytokens=[word.lemma_.lower().strip() if word.lemma_ !="_PRON_" else word.lower_ for word in mytokens]
    my_tokens=[word for word in mytokens if word not in stop_words and word not in punctuations]
    return my_tokens
class predictors(TransformerMixin):
    def transform(self,x,**transform_params):
        return [clean_text(text) for text in x]
    def fit(self,x,y=None,**fit_params):
        return self
    def get_params(self,deep=True):
        return {}
def clean_text(text):
    return text.strip().lower()

app =Flask(__name__)
df=pd.read_csv('fake_job_postings.csv')
df=df.drop(['job_id','telecommuting','has_questions','employment_type','salary_range','has_company_logo'],axis='columns')
df.fillna('',inplace=True)
def country(location):
    l=location.split(',')
    return l[0]
df['country']=df.location.apply(country)
df['text']=df['title']+''+df['company_profile']+''+df['description']+''+df['requirements']+''+df['benefits']
del df['title']
del df['location']
del df['company_profile']
del df['description']
del df['requirements']
del df['benefits']
del df['department']
del df['required_experience']
del df['required_education']
del df['industry']
del df['function']
del df['country']
fraudjob_text=df[df.fraudulent==1].text
realjob_text=df[df.fraudulent==0].text
punctuations=string.punctuation
nlp=spacy.load("en_core_web_sm")
stop_words=spacy.lang.en.stop_words.STOP_WORDS
parser=English()
df['text']=df['text'].apply(clean_text)
cv=TfidfVectorizer(max_features=100)
x=cv.fit_transform(df['text'])
df1=pd.DataFrame(x.toarray(),columns=cv.get_feature_names())
df.drop(['text'],axis=1,inplace=True)
main_df=pd.concat([df1,df],axis=1)
main_df.drop(main_df[main_df["fraudulent"]==''].index,inplace=True)
main_df["fraudulent"] = main_df["fraudulent"].map(lambda x: float(x))
y=main_df.iloc[:,-1]
x=main_df.iloc[:,:-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
z=RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=100,criterion='entropy')
model=z.fit(x_train,y_train)
predicted=z.predict(x_test)
score=accuracy_score(y_test,predicted)


@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=pd.DataFrame(cv.transform(data).toarray())
        mypred=z.predict(vect)
        return render_template('result.html',prediction=mypred)
if __name__=='__main__':
    app.run(port=4000)"""
from flask import Flask , render_template, url_for,request
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file 
import matplotlib.pyplot as plt
import string
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
app =Flask(__name__)
data=pd.read_csv('fake_job_postings.csv')
data.drop(['job_id' , 'salary_range' , 'telecommuting' , 'has_company_logo' , 'has_questions'] , axis = 1,inplace = True)
data.fillna(' ',inplace=True)
def split(location):
  l =location.split(',')
  return l[0]

data['country'] =data.location.apply(split) 
data['text'] = data['title'] + ' '+ data['location'] + ' ' + data['department'] + ' ' + data['company_profile'] + ' '+ data['description'] + ' ' + data['requirements'] + ' ' + data['benefits'] + ' ' + data['industry']

del data['title']
del data['location']
del data['department']
del data['company_profile']
del data['description']
del data['requirements']
del data['benefits']
del data['employment_type']
del data['required_experience']
del data['required_education']
del data['industry']
del data['function']
del data['country']
data.drop(data[data['fraudulent']==' '].index, inplace = True)

stop_words = set(stopwords.words("english"))
data['text'] = data['text'].apply(lambda x:x.lower())
data['text'] = data['text'].apply(lambda x:' '.join([word for word in x.split() if word not in(stop_words)]))
from sklearn.model_selection import train_test_split

X_train,X_test ,y_train,y_test = train_test_split(data.text, data.fraudulent ,test_size =0.3)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
from sklearn.naive_bayes import  MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score ,confusion_matrix ,classification_report
X_test_dtm = vect.transform(X_test)
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train.astype(int))
y_pred_nb = nb.predict(X_test_dtm)
y_test=y_test.astype('int')
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        v = vect.transform(data)
        mypred=nb.predict(v)
        return render_template('result.html',prediction=mypred)
if __name__=='__main__':
    app.run(port=4000)