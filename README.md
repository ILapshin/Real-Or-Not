# Real-Or-Not
This is my machine learning project for participating in the Kaggle Real or Not? NLP with Disaster Tweets competition.

Two ways for preparing data were used. First one was a bag of words with and without using TF-IDF and with and without n-grams. Second one was a word2vec representation with a GloVe pre-trained twitter model. Before all these methods were apllied text was cleaned and lemmatized.

Classification was made with a Decision Tree model, a k-Nearest Neighbours model, a Ridge classification model and with a neural network. The best result of 0.80265 accuracy was recieved with a Ridge classification of TF-IDF one- and bi-grams data.
