import pandas as pd 
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.util import bigrams, ngrams, everygrams
import re

# These dictionaries must be downloaded only once:
# nltk.download('wordnet') 
# nltk.download('stopwords')

def stem(series, stop_words=True):
    # Stems each sentence word-wise, returns a stemmed series
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))    
    for i, item in enumerate(series):
        temp = []
        for word in nltk.word_tokenize(item):
            if stop_words:
                if not word in stop_words:
                    temp.append(stemmer.stem(word)) 
            else:
                temp.append(stemmer.stem(word)) 
        series[i] = (' '.join(temp))
    return series

def lemm(series, stop_words=True):
    # Lemmatizes each sentence word-wise, returns a lemmatized series
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))    
    for i, item in enumerate(series):
        temp = []
        for word in nltk.word_tokenize(item):
            if stop_words:
                if not word in stop_words:
                    temp.append(lemmatizer.lemmatize(word)) 
            else:
                temp.append(lemmatizer.lemmatize(word)) 
        series[i] = (' '.join(temp))   
    return series 

def get_ngrams(series, dim):
    # Creates an array of n-grams from a series of sentences
    result = []
    for item in series:
        result.append(list(ngrams(nltk.word_tokenize(item), dim)))
    return result

def output(id, data, file_name):
    # Outputs a *.csv file ready for Kaggle application
    output = pd.DataFrame({'id': id, 'target': data})
    output.to_csv(file_name + '.csv', index=False)

def clean(series):
    # Removes noise patterns and signs from each item of a series
    series = series.str.replace(r'http://\S+|https://\S+|@\S+|#|[^A-Za-z\'\s{1}]|\'\S', '')
    series = series.str.lower()
    return series