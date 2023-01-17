from fastapi import FastAPI
import pickle
from preprocess import tokenize
import logging
import time


app = FastAPI()
# loading tfidf model
tfidf = pickle.load(open('tfidf.pickle', 'rb'))
rl_model = pickle.load(open('rl_model.pkl', 'rb'))

# it is the basic configuration it will create stream handler
logging.basicConfig(filename='logfile.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')



@app.get('/')
def home():
    return {'welcome to cyber security classification model'}


@app.post('/cyber_security_prediction/')
def predict(text):

    start = time.perf_counter()
    #  proprocess text
    clean_text = tokenize(text)
    logging.info('preprocess done')

    # converting text into numbers
    vector=tfidf.transform([clean_text])
    logging.info('converted text into numbers')


    # predict the probability 
    prediction = rl_model.predict_proba(vector)
    pred=prediction[:,1]
    logging.info(f'prediction value is {pred}')


    if pred <= 0.5:
        msg = 'not cyber security tweets'
    else:
        msg = 'cyber security tweets'

    logging.info(f'text is {msg}')
    end = time.perf_counter()
    logging.info(f'finished time is {end}-{start}')

    return {'predicted message is': msg,
    'probability' : float(pred*100)}