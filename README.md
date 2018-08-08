# SCKR-Course
Peking University Semantic Computation and Knowledge Retrieval Course Project

## `` WordSimilarity ``

Calculate the word similarity using different methods

* ``data``

	MTURK-771.csv: Data set with the groud truth.

	text8: The corpus from Wikipedia for training the Word2vec.(Please add glove.6B.300d.txt in dataset youself.)

* ``result``
	
	``corpus``: The result and model of word2vec.
	
	``web_search``: The results of jaccard, overalap, pmi and dice.
	
	``wordnet``: The results of path, wup, lch, res, lin and jcn.

* `word_similarity.py`: Codes.

* `report.pdf`: The report of Course Project. 


## `` SentimentClassification ``

Tweet Sentiment Classification [(SemEval2017 Task 4 Subtask A)](http://alt.qcri.org/semeval2017/task4/index.php)

* ``data``

	glove.6B.300d：Pre-trained word vectors (dimension = 300). (Please add glove.6B.300d.txt in dataset youself.)

	twitter-2016train-A/twitter-2016dev-A/twitter-2016test-A: Tweets, divided into training set, valid set and test set.

* `code`: Codes.

* `report.pdf`: The report of Course Project. 


## `` DBQA `` 

Document-based Question Answering task [(DBQA)](http://tcci.ccf.org.cn/conference/2017/taskdata.php)

* ``data``

	hanlp-wiki-vec-zh.txt: Pre-trained word vectors (dimension = 300). (Please add hanlp-wiki-vec-zh.txt in dataset youself.)

	stop_words.txt: Chinese stop words.

	nlpcc-iccpol-2016.dbqa.training-data/nlpcc-iccpol-2016.dbqa.testing-data/test.txt: NLPCC2017DBQA data, divided into training set, valid set and test set.

* `code`

	``DBQA_CNN&Attention1``: CNN with Attention Model 1.

	``DBQA_CNN&Attention2``: CNN with Attention Model 2.

* `report.pdf`: The report of Course Project. 
