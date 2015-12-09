# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause


from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
import sys
import numpy as np
from csv import DictReader, DictWriter
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import csv
import json
from collections import defaultdict


n_features = 5000
n_topics = 2
n_top_words = 50

# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

t0 = time()
print("Loading dataset and extracting TF-IDF features...")

question = []
for row in DictReader(open('sci_train.csv')):
	question += [row['question']]

for row in DictReader(open('sci_test.csv')):
	question += [row['question']]

vectorizer = TfidfVectorizer(max_df=0.90, min_df=1, max_features=n_features,
                             stop_words='english')

tfidf = vectorizer.fit_transform(question)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

feature_names = vectorizer.get_feature_names()

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))

#	normalized probability matrix 
topic_prob = nmf.transform(tfidf)
for i in xrange(0, topic_prob.shape[0]):
	ss = sum(topic_prob[i])
	if ss > 0:
		topic_prob[i] /= ss
	else:
		for j in xrange(0, topic_prob.shape[1]):
			topic_prob[i, j] = 1.0/topic_prob.shape[1]


#	Some objects for cleaning data
punct = set(string.punctuation)
stop = stopwords.words('english')
stemmer = WordNetLemmatizer()


#	Write cleaned training data
train = defaultdict(dict)

counts = 0
for row in DictReader(open('sci_train.csv')):
	words = []
	raw = row['question'].lower()
	sent = ''.join(ch for ch in raw if ch not in punct).split()
	for w in sent:
		ws = stemmer.lemmatize(w)
		if ws not in stop:
			words.append(ws)

	if row['correctAnswer'] == 'A':
		answer = row['answerA']
	elif row['correctAnswer'] == 'B':
		answer = row['answerB']
	elif row['correctAnswer'] == 'C':
		answer = row['answerC']
	else:
		answer = row['answerD']

	train[row['id']]['words'] = words
	train[row['id']]['answer'] = answer
	for i in xrange(0, n_topics):
		train[row['id']]['topic'+str(i)] = topic_prob[counts][i]
	counts += 1

json.dump(train, open('post_train.json', 'w'))

#	write cleaned test data
test = defaultdict(dict)

for row in DictReader(open('sci_test.csv')):
	words = []
	raw = row['question'].lower()
	sent = ''.join(ch for ch in raw if ch not in punct).split()
	for w in sent:
		ws = stemmer.lemmatize(w)
		if ws not in stop:
			words.append(ws)

	test[row['id']]['words'] = words
	test[row['id']]['answerA'] = row['answerA']
	test[row['id']]['answerB'] = row['answerB']
	test[row['id']]['answerC'] = row['answerC']
	test[row['id']]['answerD'] = row['answerD']

	for i in xrange(0, n_topics):
		test[row['id']]['topic'+str(i)] = topic_prob[counts][i]
	counts += 1
json.dump(test, open('post_test.json', 'w'))
