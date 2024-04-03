from nltk.stem import PorterStemmer
import nltk
import re
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

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