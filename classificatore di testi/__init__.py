#!/usr/bin/env python
# coding: utf-8

import pandas
import glob

lista_categorie = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
dataset_categorie = ["data/sport/*.txt", "data/world/*.txt", "data/us/*.txt", "data/business/*.txt", "data/health/*.txt",
                  "data/entertainment/*.txt", "data/sci_tech/*.txt", ]

dataset = list(map(lambda x: glob.glob(x), dataset_categorie))
dataset = [item for sublist in dataset for item in sublist]

training_data = []

for t in dataset:
    f = open(t, 'r')
    f = f.read()
    t = f.split('\n')
    training_data.append({'data': t[0] + ' ' + t[1], 'flag': lista_categorie.index(t[6])})

training_data[0]
training_data = pandas.DataFrame(training_data, columns=['data', 'flag'])
training_data.to_csv("train_data.csv", sep=',', encoding='utf-8')
print(training_data.data.shape)

import pickle
from sklearn.feature_extraction.text import CountVectorizer


#GET VECTOR COUNT
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data.data)

#SAVE WORD VECTOR
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb"))

from sklearn.feature_extraction.text import TfidfTransformer

#TRANSFORM WORD VECTOR TO TF IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#SAVE TF-IDF
pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))

# Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#clf = MultinomialNB().fit(X_train_tfidf, training_data.flag)
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf = MultinomialNB().fit(X_train, y_train)

#SAVE MODEL
pickle.dump(clf, open("nb_model.pkl", "wb"))

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

lista_categorie = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]

docs_new = "Messi joins other football team"
docs_new = [docs_new]

#LOAD MODEL
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("nb_model.pkl","rb"))

X_new_counts = loaded_vec.transform(docs_new)
X_new_tfidf = loaded_tfidf.transform(X_new_counts)
predicted = loaded_model.predict(X_new_tfidf)

print(lista_categorie[predicted[0]])

predicted = loaded_model.predict(X_test)
result_bayes = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})
result_bayes.to_csv('res_bayes.csv', sep = ',')

for element, result in zip(predicted, y_test):
    print(lista_categorie[element], ' - ', lista_categorie[result])
