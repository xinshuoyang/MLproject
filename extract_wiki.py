from wikipedia import Wikipedia
from wiki2plain import Wiki2Plain
from csv import DictReader, DictWriter
import sys


AnswerSet = []
for row in DictReader(open('sci_train.csv')):
	if row['correctAnswer'] == 'A':
		answer = row['answerA']
	elif row['correctAnswer'] == 'B':
		answer = row['answerB']
	elif row['correctAnswer'] == 'C':
		answer = row['answerC']
	else:
		answer = row['answerD']
	if answer not in AnswerSet:
		AnswerSet.append(answer)

for row in DictReader(open('sci_test.csv')):
	for choice in ['answerA', 'answerB', 'answerC', 'answerD']:
		if row[choice] not in AnswerSet:
			AnswerSet.append(row[choice])


lang = 'simple'
wiki = Wikipedia(lang)

counts = 0
n_answer = 0
o = DictWriter(open("wiki.csv", 'wb'), ["answer",  "question"])
o.writeheader()

counts = 0
for answer in AnswerSet:
	print n_answer
	n_answer += 1
	try:
	    raw = wiki.article(answer)
	except:
		raw = None

	if raw:
		question = Wiki2Plain(raw).text.split('\n')[0]
		d = {'answer': answer, 'question': question}
		o.writerow(d)

	print counts
	counts += 1


# counts = 0
# total = 0
# for row in DictReader(open('sci_train.csv')):
# 	for answer_candid in ['answerA', 'answerB', 'answerC', 'answerD']:
# 		if answer_candid[-1] == row['correctAnswer']:
# 			try:
# 			    raw = wiki.article(row[answer_candid])
# 			except:
# 				raw = None

# 			if raw:
# 				wiki2plain = Wiki2Plain(raw)
# 				print wiki2plain.text.split('\n')[0]
# 				print '-'*100
# 			else:
# 				print 'None'
# 	counts += 1
# 	if counts > 3:
# 		break
		# if raw:
		#     wiki2plain = Wiki2Plain(raw)
		#     content = wiki2plain.text
		#     print type(content)
		#     print content[:10]