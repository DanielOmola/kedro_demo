import pandas as pd
import numpy as np
from numpy import random
# import gensim
import nltk
# nltk.download()
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download("stopwords") 

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()




from nltk.corpus import stopwords
import re


REPLACE_BY_SPACE_RE = re.compile('''[/(){}\[\]\|@,;]''')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
REPLACE_OMOJI_RE = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U0001F1F2-\U0001F1F4"  # Macau flag
            u"\U0001F1E6-\U0001F1FF"  # flags
            u"\U0001F600-\U0001F64F"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U0001F1F2"
            u"\U0001F1F4"
            u"\U0001F620"
            u"\u200d"
            u"\u2640-\u2642"
            "]+", flags=re.UNICODE)

SW = []
english_sw = stopwords.words('english')
SW.extend(english_sw)
french_sw = stopwords.words('french')
# SW.extend(french_sw)

STOPWORDS = set(SW)

def clean(text:str)->str:
    """
        text: a string
        
        return: modified initial string
    """
    

    # text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = re.sub('@[A-Za-z0–9]+', '', text)
    text = re.sub('RT[\s]+', '', text)
    text = re.sub('https?:\/\/\S+', '', text)
    text = re.sub('#', '', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(' ', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = REPLACE_OMOJI_RE.sub(' ', text)
    text = re.sub(r'[0-9]+','',text)
    text = re.sub(r'[^0-9a-z #+_]','',text)
    text = re.sub(r'[/(){}\[\]\|@,;]','',text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

def lemmatize_text(text:str)->list:
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

def clean_tweeter_data(tweeter_raw_data:pd.DataFrame)->pd.DataFrame:
    tweeter_raw_data.text = tweeter_raw_data.text.apply(clean)
    return tweeter_raw_data

