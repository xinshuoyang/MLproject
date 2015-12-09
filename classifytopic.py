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

#	training data
train_question = []
train_id = []
train_answer = []
n_train = 0
for row in DictReader(open('sci_train.csv')):
	train_question += [row['question']]
	train_id += [row['id']]

	if row['correctAnswer'] == 'A':
		train_answer += [row['answerA']]
	elif row['correctAnswer'] == 'B':
		train_answer += [row['answerB']]
	elif row['correctAnswer'] == 'C':
		train_answer += [row['answerC']]
	else:
		train_answer += [row['answerD']]
	n_train += 1

#	test data
test_question = []
test_id = []
test_A = []
test_B = []
test_C = []
test_D = []
n_test = 0
for row in DictReader(open('sci_test.csv')):
	test_question += [row['question']]
	test_id += [row['id']]
	test_A += [row['answerA']]
	test_B += [row['answerB']]
	test_C += [row['answerC']]
	test_D += [row['answerD']]
	n_test += 1

vectorizer = TfidfVectorizer(max_df=0.90, min_df=1, max_features=n_features,
                             stop_words='english')

tfidf = vectorizer.fit_transform(train_question+test_question)
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
for j in xrange(0, n_train):
	words = []
	raw = train_question[j].lower()
	sent = ''.join(ch for ch in raw if ch not in punct).split()
	for w in sent:
		ws = stemmer.lemmatize(w)
		if ws not in stop:
			words.append(ws)
	answer = train_answer[j]
	train[train_id[j]]['words'] = words
	train[train_id[j]]['answer'] = answer
	for i in xrange(0, n_topics):
		train[train_id[j]]['topic'+str(i)] = topic_prob[counts][i]
	counts += 1

json.dump(train, open('post_train_v2.json', 'w'))

#	write cleaned test data
test = defaultdict(dict)
print counts


for j in xrange(0, n_test):
	words = []
	raw = test_question[j].lower()
	sent = ''.join(ch for ch in raw if ch not in punct).split()
	for w in sent:
		ws = stemmer.lemmatize(w)
		if ws not in stop:
			words.append(ws)

	test[test_id[j]]['words'] = words
	test[test_id[j]]['answerA'] = test_A[j]
	test[test_id[j]]['answerB'] = test_B[j]
	test[test_id[j]]['answerC'] = test_C[j]
	test[test_id[j]]['answerD'] = test_D[j]

	for i in xrange(0, n_topics):
		test[test_id[j]]['topic'+str(i)] = topic_prob[counts][i]
	counts += 1
json.dump(test, open('post_test_v2.json', 'w'))
