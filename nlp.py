import nltk
# nltk.download()

from nltk.tokenize import sent_tokenize, word_tokenize
EXAMPLE_TEXT = "I really hate donuts, I need burgers"

tokened_sent = sent_tokenize(EXAMPLE_TEXT)
tokened_words = word_tokenize(EXAMPLE_TEXT)
print("Tokenized sentence : ",tokened_sent)
print("Tokenized words : ",tokened_words)

#stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
for w in tokened_words:
 print(ps.stem(w))

 #Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
for w in tokened_words:
 print(lemmatizer.lemmatize(w))

 #stopwords
 from nltk.corpus import stopwords
 stop_words = set(stopwords.words('english'))
 sentence = []
 for w in tokened_words:
    if w not in stop_words:
        sentence.append(w)
print("After removing stopwords:", sentence)
