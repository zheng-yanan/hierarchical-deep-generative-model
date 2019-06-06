# -*- coding:utf-8 -*-

""" Utilities for Data Operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import sklearn

import numpy as np
import tensorflow as tf

pyVersion = sys.version_info[0]

class BaseDataLoader(object):
	def __init__(self, batch_size):
		self.train_inputs = None
		self.train_targets = None
		self.dropout_train_inputs = None

		self.valid_inputs = None
		self.valid_targets = None

		self.test_inputs = None
		self.test_targets = None

		self.word_to_index = None
		self.index_to_word = None

		self.batch_size = batch_size

	def train_generator(self):
		for i in range(0, self.train_size, self.batch_size):
			yield (self.train_inputs[i : i + self.batch_size],
				   self.train_targets[i : i + self.batch_size],
				   self.dropout_train_inputs[i : i + self.batch_size])


	def valid_generator(self):
		for i in range(0, self.valid_size, self.batch_size):
			yield (self.valid_inputs[i : i + self.batch_size],
				   self.valid_targets[i : i + self.batch_size])

	def test_generator(self):
		for i in range(0, self.test_size, self.batch_size):
			yield (self.test_inputs[i : i + self.batch_size],
				   self.test_targets[i : i + self.batch_size])


class IMDB(BaseDataLoader):
	def __init__(self, batch_size, seq_len, word_dropout_rate, log_manager):
		BaseDataLoader.__init__(self, batch_size)
		self.seq_len = seq_len
		self.word_dropout_rate = word_dropout_rate		

		self._index_from = 4
		self.word_to_index, self.index_to_word = self._build_vocab()
		self.UNK_ID = self.word_to_index["<unk>"]
		self.EOS_ID = self.word_to_index["<eos>"]
		self.PAD_ID = self.word_to_index["<pad>"]
		self.SOS_ID = self.word_to_index["<sos>"]
		log_manager.info("Vocabulary Loaded.")

		self.train_inputs, self.train_targets, self.test_inputs, self.test_targets = self._load_data()
		log_manager.info("IMDB Data Loaded.")

		self.dropout_train_inputs = self.dropout_data()
		log_manager.info("Train Inputs Dropout.")		

		self.train_size = len(self.train_inputs)
		self.test_size = len(self.test_inputs)
		self.vocab_size = len(self.word_to_index)
		log_manager.info("IMDB Statistics:")
		log_manager.info("Train Set Size: %d" % self.train_size)
		log_manager.info("Test Set Size: %d" % self.test_size)
		log_manager.info("Vocabulary Size: %d" % self.vocab_size)


	def _build_vocab(self):
		word_to_index = tf.contrib.keras.datasets.imdb.get_word_index()
		word_to_index = {w: (i + self._index_from) for w,i in word_to_index.items()}
		word_to_index["<pad>"] = 0
		word_to_index["<sos>"] = 1
		word_to_index["<unk>"] = 2
		word_to_index["<eos>"] = 3
		index_to_word = {i:w for (w,i) in word_to_index.items()}
		index_to_word[-1] = "-1"
		return word_to_index, index_to_word


	def _load_data(self):
		(train, _), (test, _) = tf.contrib.keras.datasets.imdb.load_data(num_words=None, index_from = self._index_from)
		train_inputs, train_targets = self._pad(train)
		test_inputs, test_targets = self._pad(test)
		return train_inputs, train_targets, test_inputs, test_targets


	def _pad(self, data):
		inputs = []
		targets = []
		for ins in data:
			if len(ins) < self.seq_len - 1:
				inputs.append([self.SOS_ID] + ins + [self.PAD_ID] * (self.seq_len - 1 - len(ins)))
				targets.append(ins + [self.EOS_ID] + [self.PAD_ID] * (self.seq_len - 1 - len(ins)))
			else:
				truncated = ins[:(self.seq_len - 1)]
				inputs.append([self.SOS_ID] + truncated)
				targets.append(truncated + [self.EOS_ID])

				truncated = ins[-(self.seq_len-1):]
				inputs.append([self.SOS_ID] + truncated)
				targets.append(truncated + [self.EOS_ID])
		return np.array(inputs), np.array(targets)		

	# if word_dropout_rate == 0, no words are dropped.
	def _word_dropout(self, inputs):
		is_dropped = np.random.binomial(1, self.word_dropout_rate, inputs.shape)
		fn = np.vectorize(lambda inputs, k: self.UNK_ID if k else inputs)
		return fn(inputs, is_dropped)


	def dropout_data(self):
		return self._word_dropout(self.train_inputs)


	def shuffle(self):
		self.train_inputs, self.train_targets, self.dropout_train_inputs = sklearn.utils.shuffle(self.train_inputs, self.train_targets, self.dropout_train_inputs)
		self.test_inputs, self.test_targets = sklearn.utils.shuffle(self.test_inputs, self.test_targets)



class PTB(BaseDataLoader):
	def __init__(self, batch_size, seq_len, word_dropout_rate, log_manager):
		BaseDataLoader.__init__(self, batch_size)


		_base_path = "data/ptb/simple-examples/data"
		self.seq_len = seq_len
		self.word_dropout_rate = word_dropout_rate

		self.word_to_index, self.index_to_word = self._build_vocab(os.path.join(_base_path,"ptb.train.txt"))
		self.UNK_ID = self.word_to_index["<unk>"]
		self.EOS_ID = self.word_to_index["<eos>"]
		self.PAD_ID = self.word_to_index["<pad>"]
		self.SOS_ID = self.word_to_index["<sos>"]
		log_manager.info("Vocabulary Loaded.")

		self.train_inputs, self.train_targets = self._load_data(os.path.join(_base_path, "ptb.train.txt"))
		self.valid_inputs, self.valid_targets = self._load_data(os.path.join(_base_path, "ptb.valid.txt"))
		self.test_inputs, self.test_targets = self._load_data(os.path.join(_base_path, "ptb.test.txt"))
		log_manager.info("PTB Data Loaded.")

		self.dropout_train_inputs = self.dropout_data()
		log_manager.info("Train Inputs Dropout.")

		self.train_size = len(self.train_inputs)
		self.valid_size = len(self.valid_inputs)
		self.test_size = len(self.test_inputs)
		self.vocab_size = len(self.word_to_index)
		log_manager.info("PTB Statistics:")
		log_manager.info("Train Set Size: %d" % self.train_size)
		log_manager.info("Valid Set Size: %d" % self.valid_size)
		log_manager.info("Test Set Size: %d" % self.test_size)
		log_manager.info("Vocabulary Size: %d" % self.vocab_size)

	def _read_words(self, filename):
		with tf.gfile.GFile(filename, "r") as f:
			if pyVersion == 3:
				return f.read().replace("\n", "<eos>").split()
			else:
				return f.read().decode("utf-8").replace("\n", "<eos>").split()


	def _build_vocab(self, filename):
		data = self._read_words(filename)
		counter = collections.Counter(data)
		count_pairs = counter.most_common()
		words, _ = list(zip(*count_pairs))
		word_to_index = dict(zip(words, range(len(words))))
		word_to_index = {w: (i + 2) for (w, i) in word_to_index.items()}
		word_to_index["<pad>"] = 0
		word_to_index["<sos>"] = 1
		index_to_word = {i: w for (w, i) in word_to_index.items()}
		index_to_word[-1] = "-1"
		return word_to_index, index_to_word

	def _load_data(self, filename):
		words = self._read_words(filename)
		raw_data = [self.word_to_index[word] for word in words if word in self.word_to_index]

		data_len = len(raw_data)
		batch_len = data_len // self.batch_size
		data = np.reshape(raw_data[0:self.batch_size * batch_len], [self.batch_size, batch_len])
		epoch_size = (batch_len - 1) // self.seq_len

		inputs = []
		targets = []
		for i in range(epoch_size):
			start_index = i * self.seq_len
			end_index = (i + 1) * self.seq_len
			inputs.append(data[:, start_index : end_index])
			targets.append(data[:, start_index + 1 : end_index + 1])

		return np.concatenate(inputs), np.concatenate(targets)


	# if word_dropout_rate == 0, no words are dropped.
	def _word_dropout(self, inputs):
		is_dropped = np.random.binomial(1, self.word_dropout_rate, inputs.shape)
		fn = np.vectorize(lambda inputs, k: self.UNK_ID if k else inputs)
		return fn(inputs, is_dropped)


	def dropout_data(self):
		return self._word_dropout(self.train_inputs)


	def shuffle(self):
		self.train_inputs, self.train_targets, self.dropout_train_inputs = sklearn.utils.shuffle(self.train_inputs, self.train_targets, self.dropout_train_inputs)
		self.valid_inputs, self.valid_targets = sklearn.utils.shuffle(self.valid_inputs, self.valid_targets)
		self.test_inputs, self.test_targets = sklearn.utils.shuffle(self.test_inputs, self.test_targets)