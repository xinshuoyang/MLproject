from csv import DictReader, DictWriter
from collections import defaultdict
from sets import Set
import sys
import json
from sklearn.feature_extraction.text import TfidfVectorizer

#	Get train data
train = defaultdict(list)
for row in DictReader(open('sci_train.csv')):
	if row['correctAnswer'] == 'A':
		train[row['answerA']] += [row['question']]
	elif row['correctAnswer'] == 'B':
		train[row['answerB']] += [row['question']]
	elif row['correctAnswer'] == 'C':
		train[row['answerC']] += [row['question']]
	else:
		train[row['answerD']] += [row['question']]

for ii in train.keys():
	train[ii] = [' '.join(train[ii])]

#	Get test data
test = defaultdict(dict)
for row in DictReader(open('sci_test.csv')):
	test[row['id']]['answerA'] = row['answerA']
	test[row['id']]['answerB'] = row['answerB']
	test[row['id']]['answerC'] = row['answerC']
	test[row['id']]['answerD'] = row['answerD']
	test[row['id']]['question'] = row['question']



o = DictWriter(open("predictions.csv", 'wb'), ["id",  "correctAnswer"])
o.writeheader()
for ii in test.keys():
	documents = []
	answer = []
	for choice in ['answerA', 'answerB', 'answerC', 'answerD']:
		if test[ii][choice] in train.keys():
			documents += train[test[ii][choice]]
			answer.append(choice)
	documents += [test[ii]['question']]
	vect = TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1, 2))
	tfidf = vect.fit_transform(documents)
	pred = answer[(tfidf * tfidf.T).A[-1][:-1].argsort()[-1]][-1]
	d = {'id': ii, 'correctAnswer': pred}
	o.writerow(d)