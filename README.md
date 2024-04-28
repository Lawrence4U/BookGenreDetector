# Book Genre Detector using traditional ML Models as well as Deep Learning and NLP

## Preprocessing

Two datasets were found for the development of the project: BooksDataSet.csv from [GitHub](https://github.com/chikne97/Book-Genre-Prediction/tree/master) and booksummaries.txt from [CMU](https://www.cs.cmu.edu/~dbamman/booksummaries.html). The first dataset is a preprocessed version of the second.

In a first pass through the models, the first model resulted incomplete to be used for deep learning for having only 500 samples per category, so some processing was made in the first dataset to add samples.

After adding the extra samples the resulting dataset comes as follows:

![alt text](imgs\image.png)

Each summary requires some preprocessing to be able to be sent to any classification model. For this, some basic NLP processing is done. For our case the process is as follows:
1. Tokenization of sentences
2. Filter of special characters
3. Tokenization of words
4. Stemming and lemmatization of words
5. removal of stop_words

After this first iteration we obtain the following preliminary results:
![alt text](imgs\image-1.png)
We can see that there are still several words that are undesirable like "hi" and "ha" which provide no real meaning in determining the significance in the summary.

Plotting these per genre in a wordcloud we obtain the following:
![alt text](imgs\image-2.png)
Because of this, the next step was decided:

6. **Manual cleansing of other undesired words**

This process is iterative sindce when removing some of the most frequent words then others take their place and may or may not be important fot determining their importance in the classification. After removing the most present words that don't have any significance, we get the following plots:
![alt text](imgs/image3.png)
![alt text](imgs/output.png)

There is a clear tendency for some words in english literature with words like "kill", time and new being one of the most repeated. Anyhow, we can now see better some words that are genre defining. One the most clear is science fiction where we can find words like planet, human, alien and such which are common topics discussed in those kinds of novels. THe same happens with others like Crime Fiction where kill, murder and investigate are present really frequently.

## ML Models

Just for the sake of determining if the extra manual processing was benefitial, each model has been run with the regular processing from steps 1 to 5 and again with step 6 included and we will compare accuracies.




## DeepLearning Models



Started with BooksDataSet but too low data 
High Precision on Models