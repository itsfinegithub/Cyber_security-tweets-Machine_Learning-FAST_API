
import numpy as np
import re
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
STOPWORDS=set(stopwords.words('english'))
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer= WordNetLemmatizer()



def tokenize(text):
    text = re.sub(r'http\S+', '', text) #removing links

    text = re.sub("[^a-zA-Z]", " ", text.lower())#panctuations
    
    tokens = word_tokenize(text) #split text into words
    
    words = [w for w in tokens if w not in STOPWORDS] #remove stopwords
    
    lemmed_words = [lemmatizer.lemmatize(w) for w in words] #lemmatizing
    
    clean_tokens = []
    
    for i in lemmed_words: #appending all the lemmed words to clean_tokens
        clean_tokens.append(i)
        
    text = " ".join(clean_tokens) #joining text in cleaned tokens based on white spaces
    
    return text