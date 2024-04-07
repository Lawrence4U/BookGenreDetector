import pandas as pd
import nltk
import re
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter

def get_stems(words: list):
    ps = PorterStemmer()
    stems = []
    for word in words:
        stems.append(ps.stem(word))
    return stems

def get_lemma(words: list):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma = []
    for word in words:
        lemma.append(wordnet_lemmatizer.lemmatize(word)) # Warning: Lemmatizer needs a POS tag or else it treats it as a noun and doesn't change it
    return(lemma)

# Create a set of stop words 
stop_words = set(stopwords.words('english'))
def remove_stop_words(words: list):
    return [w for w in words if not w.lower() in stop_words]

def get_sent_tokens(data):
    sentences = sent_tokenize(data)
    return sentences

def get_word_tokens(sentences):
    words = []
    for sent in sentences:
        words.extend(word_tokenize(sent))
    return(words)

def remove_special_characters(sentences):
    clean_sentences = []
    for sent in sentences:
        pattern = r'[^a-zA-Z0-9\s]'
        clean_text = re.sub(pattern, '', sent)
        clean_sentences.append(clean_text)
    return clean_sentences

# gets word frequency and returns a dict with top max_features elements
def get_word_counter(column, max_features=None):
    all_words = ' '.join(column).split(' ')
    word_count = Counter(all_words)
    
    return dict(word_count.most_common(max_features))

# Gets a list and returns the words turned into assigned integers
def filter_and_tokenize_words(words: list, word_dict: dict):
    result = []
    for word in words:
        if word in word_dict:
            result.append(word_dict[word])
            
    return result

def tokenize_summary(row):
    return get_word_tokens(get_sent_tokens(row))
    
    