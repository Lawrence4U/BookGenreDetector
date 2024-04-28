# import pandas as pd
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
# import numpy as np
# from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
#from keras.losses import BinaryCrossentropy

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

# def create_model(input_dim=1000, num_classes=9, optimizer='adam', 
#                  criterion="categorical_crossentropy", activation='relu', dropout_rate=0.0,
#                  neurons=64,  layers=3, learning_rate=0.001):
#     optimizer = build_optimizer(optimizer, learning_rate=learning_rate)
#     criterion = build_criterion(criterion)
    
#     model = Sequential()
#     for _ in range(layers):
#         model.add(Dense(neurons, input_dim=input_dim, activation=activation))
#         model.add(Dropout(dropout_rate))
#     model.add(Dense(neurons, activation=activation))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])
#     return model

def build_optimizer(optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = SGD(lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = Adam(lr=learning_rate)
    return optimizer

# def build_criterion(criterion):
#     return BinaryCrossentropy()
     
# def train_model(model, epochs, x_train, x_test, y_train, y_test, optimizer, criterion, device):
#     history = {'train_loss': [], 'train_accuracy': [], 'valid_loss': [], 'valid_accuracy': []}
#     for epoch in range(epochs):
#         model.train()  # Set the model to train mode
#         train_loss = 0.0
#         train_accuracy = 0.0
        
#         # Iterate over batches of training data
#         for inputs, labels in zip(x_train, y_train):
#             # Zero the parameter gradients
#             optimizer.zero_grad()
#             # Forward pass
#             inputs = inputs.unsqueeze(0)  # Add batch dimension
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
            
#             # Compute loss
#             loss = criterion(outputs, labels.unsqueeze(0).float())
            
#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
            
#             # Update running loss
#             train_loss += loss.item()
#             train_accuracy += (outputs.argmax(1) == labels).sum().item()
        
#         train_loss /= len(x_train)
#         train_accuracy /= len(x_train)
#         history['train_loss'].append(train_loss)
#         history['train_accuracy'].append(train_accuracy)
        
#         print(f'Epoch {epoch + 1}/{epochs} - '
#             f'Train Loss: {train_loss:.4f}, '
#             f'Train Accuracy: {train_accuracy:.4f}')
        
#         model.eval()
#         valid_loss = 0.0
#         valid_accuracy = 0.0
#         for inputs, labels in zip(x_test, y_test):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             valid_loss += loss.item()
#             valid_accuracy += (outputs.argmax(1) == labels).sum().item()

#         valid_loss /= len(x_test)
#         valid_accuracy /= len(x_test)
#         history['valid_loss'].append(valid_loss)
#         history['valid_accuracy'].append(valid_accuracy)

#         print(f'Epoch {epoch + 1}/{epochs} - '
#                 f'Validation Loss: {valid_loss:.4f}, '
#                 f'Validation Accuracy: {valid_accuracy:.4f}')
#     return history