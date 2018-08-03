import gensim
import numpy as np
import jieba

class Word2Vec():
	def __init__(self):
		self.model = gensim.models.KeyedVectors.load_word2vec_format('data/hanlp-wiki-vec-zh.txt', binary=False)
		self.unknowns = np.random.uniform(-0.01, 0.01, 300).astype("float32")

	def get(self, word):
		if word in self.model.vocab:
			return self.model.word_vec(word)
		else:
			return self.unknowns		


def seg_sentence(sentence):
	segment = jieba.cut(sentence.strip())

	segment = [word for word in segment]

	return segment
	

class QA_pair():
	def __init__(self, file_path, sentence_length):
		self.questions, self.answers, self.labels, self.features = [], [], [], []
		self.word2vec = Word2Vec()
		self.index = 0
		self.max_len = 0
		self.sentence_length = sentence_length

		with open(file_path, 'r', encoding='utf-8') as f:
			stop_words = [line.strip() for line in open('data/stop_words.txt', 'r', encoding='utf-8').readlines()]

			for line in f:
				temp = line.split('\t')
				question = seg_sentence(temp[0])
				answer = seg_sentence(temp[1])
				label = int(temp[2])

				self.questions.append(question)
				self.answers.append(answer)
				self.labels.append(label)
				word_cnt = len([word for word in question if (word not in stop_words) and (word in answer)])
				self.features.append([len(question), len(answer), word_cnt])

				max_len = max(len(question), len(answer))
				if max_len > self.max_len:
					self.max_len = max_len				

		self.data_size = len(self.questions)

		flatten = lambda l : [item for sublist in l for item in sublist]
		question_vocab = list(set(flatten(self.questions)))
		idf = {}
		for w in question_vocab:
			idf[w] = np.log(self.data_size / len([1 for question in self.questions if w in question]))

		for i in range(self.data_size):
			wgt_word_cnt = sum([idf[word] for word in self.questions[i] if (word not in stop_words) and (word in self.answers[i])])
			self.features[i].append(wgt_word_cnt)

		self.num_features = len(self.features[0])


	def reset_index(self):
		self.index = 0


	def is_available(self):
		if self.index < self.data_size:
			return True
		else:
			return False


	def next_batch(self, batch_size):
		batch_size = min(self.data_size - self.index, batch_size)
		question_mats, answer_mats = [], []

		for i in range(batch_size):
			question = self.questions[self.index + i]
			answer = self.answers[self.index + i]

			# [1, dimesion, sentence_length]
			if len(question) > self.sentence_length:
				question = question[:self.sentence_length]

			if len(answer) > self.sentence_length:
				answer = answer[:self.sentence_length]

			question_mats.append(np.expand_dims(np.pad(np.column_stack([self.word2vec.get(w) for w in question]), 
				[[0, 0], [0, self.sentence_length-len(question)]], 'constant'), axis=0))

			if len(answer) == 0:
				answer_mats.append(np.expand_dims(np.pad(np.expand_dims(np.pad([], [0, self.sentence_length-len(answer)], 'constant'), axis=0), 
					[[0, 299], [0, 0]], 'constant'), axis=0))
			else:
				answer_mats.append(np.expand_dims(np.pad(np.column_stack([self.word2vec.get(w) for w in answer]), 
				[[0, 0], [0, self.sentence_length-len(answer)]], 'constant'), axis=0))

		batch_questions = np.concatenate(question_mats, axis=0)
		batch_answers = np.concatenate(answer_mats, axis=0)
		batch_labels = self.labels[self.index: self.index + batch_size]
		batch_features = self.features[self.index: self.index + batch_size]

		self.index += batch_size

		return batch_questions, batch_answers, batch_labels, batch_features


# if __name__ == '__main__':
# 	# word2vec = Word2Vec()

# 	# print(word2vec.get("我们"))

# 	# stop_words = [line.strip() for line in open('data/stop_words.txt', 'r', encoding='utf-8').readlines()]


# 	# print(len(stop_words), stop_words[0])


# 	print(seg_sentence(u'腾讯控股有限公司，简称腾讯，是一家民营IT企业，总部位于中国广东深圳，于2004年6月16日在香港交易所上市'))
# 	train_data = QA_pair('data/qa.training-data')


