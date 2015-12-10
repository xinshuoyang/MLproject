from wikipedia import Wikipedia
from wiki2plain import Wiki2Plain
from csv import DictReader, DictWriter

lang = 'simple'
wiki = Wikipedia(lang)

counts = 0
total = 0
for row in DictReader(open('sci_train.csv')):
	for answer_candid in ['answerA', 'answerB', 'answerC', 'answerD']:
		try:
		    raw = wiki.article(row[answer_candid])
		except:
			raw = None

		if raw:
			wiki2plain = Wiki2Plain(raw)
			print wiki2plain.text.split('\n')[0]
			print '-'*100
		else:
			print 'None'
	counts += 1
	if counts > 2:
		break
		# if raw:
		#     wiki2plain = Wiki2Plain(raw)
		#     content = wiki2plain.text
		#     print type(content)
		#     print content[:10]