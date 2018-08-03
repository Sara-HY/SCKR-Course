# coding = utf-8
import os
import csv
import shutil
import numpy as np
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
from scipy import stats
from gensim.models import word2vec
import requests
from bs4 import BeautifulSoup
import math
import matplotlib.pyplot as plt
	 

# load word pairs and scores 
def load_data(file_path):
	word_list = []
	score_list = []
	with open(file_path) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			word_list.append(row[0:2])
			score_list.append(row[2])
			
	return word_list, score_list


# wordnet
# Path_based Method, Wu&Palmer Method, Leacock&Chodorow Method, Reslink Method, Lin's Method, Jiang-Conrath Method
def get_similarity_by_wordnet(file_path, word_list, result_paths):
	shutil.rmtree(file_path) 
	os.mkdir(file_path)

	brown_ic = wordnet_ic.ic('ic-brown.dat')
	semcor_ic = wordnet_ic.ic('ic-semcor.dat')

	for (word1, word2) in word_list:
		synsets1 = wordnet.synsets(word1)
		synsets2 = wordnet.synsets(word2)

		pred = [-100, -100, -100, -100, -100, -100]

		for synset1 in synsets1:
			for synset2 in synsets2:
				if synset1.pos() == synset2.pos():
					try:
						pred[0] = max(pred[0], synset1.path_similarity(synset2))
					except Exception as e:
						print("E:", synset1, synset2, str(e))

					try:
						pred[1] = max(pred[1], synset1.wup_similarity(synset2))
					except Exception as e:
						print("E:", synset1, synset2, str(e))

					try:
						pred[2] = max(pred[2], synset1.lch_similarity(synset2))
					except Exception as e:
						print("E:", synset1, synset2, str(e))

					try:
						pred[3] = max(pred[3], synset1.res_similarity(synset2, brown_ic))
					except Exception as e:
						print("E:", synset1, synset2, str(e))

					try:
						pred[4] = max(pred[4], synset1.lin_similarity(synset2, brown_ic))
					except Exception as e:
						print("E:", synset1, synset2, str(e))

					try:
						pred[5] = max(pred[5], synset1.jcn_similarity(synset2, brown_ic))
					except Exception as e:
						print("E:", synset1, synset2, str(e))

		for i in range(len(pred)):
			with open(result_paths[i], 'a+') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow((word1, word2, pred[i]))
	

# word2vec wikipedia cosine
def get_similarity_by_corpus(file_path, word_list):
	shutil.rmtree(file_path) 
	os.mkdir(file_path)

	sentences = word2vec.Text8Corpus("./data/text8")
	print("trainning....")
	model = word2vec.Word2Vec(sentences)
	model.wv.save_word2vec_format(os.path.join(file_path, 'word2vec.model.bin'), binary=True)
	
	for (word1, word2) in word_list:
		try:
		 	similarity = model.similarity(word1, word2)
		except Exception as e:
		 	similarity = -100
		 	print("word2vec", str(e))

		with open(os.path.join(file_path, 'word2vec.csv'), 'a+') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow((word1, word2, similarity))


def google_return_count(word):
	url = 'https://www.google.com.hk/search?hl=en&q=%s' % word
	try:
		page = requests.get(url)

		soup = BeautifulSoup(page.content, 'html.parser', from_encoding='utf-8')
		newstext = soup.find(id="resultStats").text		
		temp = str(newstext).split(" ")[1].replace(",", '')
		return temp

	except Exception as e:
		print("Exception", e)
		return ''

# web search : page-count-based, google similarity 
def get_similarity_by_seach(file_path, word_list, result_paths):
	shutil.rmtree(file_path) 
	os.mkdir(file_path)

	# Data From http://www.statisticbrain.com/total-number-of-pages-indexed-by-google/
	googlepages = 30000000000000

	for (word1, word2) in word_list:
		count1 = google_return_count(word1)
		count2 = google_return_count(word2)
		count_both = google_return_count(word1 + "%20" + word2)
		if (count1 and count2 and count_both):
			count1 = int(count1)
			count2 = int(count2)
			count_both = int(count_both)
			
			if count_both <= 5:
				jaccard_score = 0
				overlap_score = 0
				dice_score = 0
				pmi_score = 0
			else:
				jaccard_score = 1.0 * count_both / ( count1 + count2 - count_both)
				overlap_score = 1.0 * count_both / min(count1, count2)
				dice_score = 2.0 * count_both / (count1 + count2)
				pmi_score = math.log((1.0 * count_both * googlepages) / (count1 * count2), 2)

			scores = [jaccard_score, overlap_score, dice_score, pmi_score]
		else:
			scores = [None, None, None, None]
		for i in range(len(scores)):
			with open(result_paths[i], 'a+') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow((word1, word2, scores[i]))


def get_spearman(score_list, result_paths):
	method_type = []
	method_spearman = []
	
	for i in range(len(result_paths)):
		_, pred_list = load_data(result_paths[i])
		# pred_lists.append(pred_list)
		spearman = stats.spearmanr(score_list, pred_list)[0]

		temp = result_paths[i].split('.')[1]
		typeStr = temp.split('/')[2] + '_' +  temp.split('/')[3]

		method_spearman.append(spearman)
		method_type.append(typeStr)
		
	print(method_type, method_spearman)	
	return method_type, method_spearman


def save_results(method_type, method_spearman):
	
	plt.figure(figsize=(15, 7))
	idx = np.arange(len(method_spearman))
	plt.barh(idx, method_spearman, color="#fd7601", height=0.5)
	plt.yticks(idx, method_type)
	plt.grid(axis='x')
	 
	plt.xlabel('Spearman Rank Correlation Coefficient')
	plt.ylabel('Method')
	plt.title('Word Similarity / MTURK-771')
	 
	
	plt.savefig("./report/result.png")
	plt.show()



if __name__ == '__main__':

	word_list, score_list = load_data("./data/MTURK-771.csv")
	result_paths = []

	file_path = './result/wordnet'
	wordnet_paths = [os.path.join(file_path, 'path.csv'), os.path.join(file_path, 'wup.csv'), os.path.join(file_path, 'lch.csv'), os.path.join(file_path, 'res.csv'), os.path.join(file_path, 'lin.csv'), os.path.join(file_path, 'jcn.csv')]
	# get_similarity_by_wordnet('./result/wordnet', word_list, wordnet_paths)
	result_paths += wordnet_paths


	# get_similarity_by_corpus('./result/corpus', word_list)
	result_paths += [os.path.join('./result/corpus', 'word2vec.csv')]


	file_path = './result/web_search'
	search_paths = [os.path.join(file_path, 'jaccard.csv'), os.path.join(file_path, 'overlap.csv'), os.path.join(file_path, 'dice.csv'), os.path.join(file_path, 'pmi.csv')]
	# get_similarity_by_seach('./result/web_search', word_list, search_paths)
	result_paths += search_paths
	
	
	method_type, method_spearman = get_spearman(score_list, result_paths)
	save_results(method_type, method_spearman)
	
	









