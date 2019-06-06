# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from data_utils import IMDB
from data_utils import PTB
from log_utils import logger_fn
from models.vae import VAE
from models.hdgm import HDGM


tf.app.flags.DEFINE_string("dataset","ptb", "['imdb','ptb']")
tf.app.flags.DEFINE_string("model","hdgm", "['vae','hdgm']")
FLAGS = tf.app.flags.FLAGS



class PTBConfig():
	seq_len = 20
	batch_size = 20
	embedding_dim = 128
	hidden_dim = 200
	clip_norm = 5.0
	word_dropout_rate = 0.0
	num_epoch = 100
	num_layers = 2
	display_loss_step = 100
	vocab_size = -1
	rnn_type = "block"
	lr_decay = 0.5
	max_epoch = 4

	learning_rate = 0.001
	beam_width = 5
	latent_dim = 16
	anneal_max = 1.0
	anneal_bias = 26000 # 26000
	anneal_slope = 1 / 2600 # 1/2800


	bidirect = True
	attention_dim = 50


class IMDBConfig:

	seq_len = 15
	batch_size = 20
	embedding_dim = 128
	hidden_dim = 200
	clip_norm = 5.0
	word_dropout_rate = 0.0
	num_epoch = 100
	display_loss_step = 100
	vocab_size = -1
	num_layers = 2
	rnn_type = "block"
	lr_decay = 0.5
	max_epoch = 3

	learning_rate = 0.001
	beam_width = 5
	latent_dim = 16
	anneal_max = 1.0
	anneal_bias = 18000
	anneal_slope = 1/1800


def main(_):

	model_name = FLAGS.model + "_" + FLAGS.dataset
	save_path = os.path.join("save", model_name)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	checkpoint_path = os.path.join(save_path, model_name+".ckpt")
	log_file = os.path.join(save_path, model_name + ".txt")
	log_manager = logger_fn(model_name + ".txt", log_file)	


	if FLAGS.dataset == "imdb":
		hparam = IMDBConfig()
		test_sent = "i love this film and i think it is one of the best films"
		dataloader = IMDB(hparam.batch_size, hparam.seq_len, hparam.word_dropout_rate, log_manager)
	elif FLAGS.dataset == "ptb":
		hparam = PTBConfig()
		test_sent = "many money managers and some traders had already left their offices early friday afternoon"
		dataloader = PTB(hparam.batch_size, hparam.seq_len, hparam.word_dropout_rate, log_manager)
	else:
		raise ValueError("Unrecognized dataset.")
	hparam.vocab_size = dataloader.vocab_size


	if FLAGS.model == "vae":
		model = VAE(hparam, dataloader.word_to_index, dataloader.index_to_word)
	elif FLAGS.model == "hdgm":
		model = HDGM(hparam, dataloader.word_to_index, dataloader.index_to_word)
	else:
		raise ValueError("Unrecognized model.")



	
	log_manager.info("")
	log_manager.info("Model Parameter Size: %d" % model.get_parameter_size())
	log_manager.info("")

	sess = tf.Session()
	writer = tf.summary.FileWriter(save_path, sess.graph)
	sess.run(model.init)

	for epoch in range(hparam.num_epoch):
		dataloader.dropout_data()
		log_manager.info("Word Dropout Updated.")
		dataloader.shuffle()
		log_manager.info("Data Shuffled.")

		lr_decay = hparam.lr_decay ** max(epoch + 1 - hparam.max_epoch, 0.0)
		sess.run(model.lr_update, feed_dict={model.new_lr: hparam.learning_rate * lr_decay})
		log_manager.info("Epoch: %d Learning rate: %.7f" % (epoch, sess.run(model.lr)))

		best_valid_loss = 100000


		total_loss = 0.0
		total_nll_loss = 0.0
		total_kl_loss = 0.0
		total_count = 0

		for index, (inputs, targets, _) in enumerate(dataloader.train_generator()):
			outputs = model.train_session(sess, inputs, targets)
			
			total_loss += outputs["total_loss"]
			total_nll_loss += outputs["nll_loss"]
			total_kl_loss += outputs["kl_loss"]
			total_count += 1
			writer.add_summary(outputs["summary"], outputs["step"])

			if index % hparam.display_loss_step == 0:

				log_manager.info("Step: [%4d] [%2d/%2d] [%4d/%4d] perplexity: %.6f | elbo: %.6f | nll_loss: %.6f | kl_loss: %.6f | kl_w: %.3f" % 
					(outputs["step"], 
					 epoch, hparam.num_epoch, 
					 index, dataloader.train_size//hparam.batch_size,
					 np.exp(total_nll_loss/total_count/hparam.seq_len),
					 total_loss/total_count,
					 total_nll_loss/total_count,
					 total_kl_loss/total_count,
					 outputs["kl_w"]
					 ))

		log_manager.info("")
		log_manager.info("Train Epoch: %d | Perplexity: %.6f | ELBO: %.6f | NLL_Loss: %.6f | KL_loss: %.6f" % 
			(epoch, 
				np.exp(total_nll_loss/total_count/hparam.seq_len), 
				total_loss/total_count,
				total_nll_loss/total_count,
				total_kl_loss/total_count))

		
		if dataloader.valid_inputs is not None:

			valid_nll_loss = 0.0
			valid_total_loss = 0.0
			valid_kl_loss = 0.0
			valid_count = 0

			for i, (inp, tar) in enumerate(dataloader.valid_generator()):
				valid_outputs = model.test_session(sess, inp, tar)

				valid_total_loss += valid_outputs["total_loss"]
				valid_nll_loss += valid_outputs["nll_loss"]
				valid_kl_loss += valid_outputs["kl_loss"]
				valid_count +=1

			log_manager.info("Valid Epoch: %d | Perplexity: %.6f | ELBO: %.6f | NLL_Loss: %.6f | KL_loss: %.6f" % 
				(epoch, 
					np.exp(valid_nll_loss/valid_count/hparam.seq_len), 
					valid_total_loss/valid_count,
					valid_nll_loss/valid_count,
					valid_kl_loss/valid_count))


			if valid_total_loss/valid_count < best_valid_loss:
				best_valid_loss = valid_total_loss/valid_count
				model.saver.save(sess, checkpoint_path)
				log_manager.info("Model saved in file: %s" % checkpoint_path)
				log_manager.info("")
		

		if dataloader.test_inputs is not None:

			test_total_loss = 0.0
			test_nll_loss = 0.0
			test_kl_loss = 0.0
			test_count = 0

			for i, (inp, tar) in enumerate(dataloader.test_generator()):
				test_outputs = model.test_session(sess, inp, tar)

				test_total_loss += test_outputs["total_loss"]
				test_nll_loss += test_outputs["nll_loss"]
				test_kl_loss += test_outputs["kl_loss"]
				test_count += 1

			log_manager.info("Test Epoch: %d | Perplexity: %.6f | ELBO: %.6f | NLL_Loss: %.6f | KL_loss: %.6f" % 
				(epoch, 
					np.exp(test_nll_loss/test_count/hparam.seq_len),
					test_total_loss/test_count,
					test_nll_loss/test_count,
					test_kl_loss/test_count
					))
			log_manager.info("")


		ret1, ret2 = model.reconstruct_session(sess, test_sent)
		log_manager.info(ret1)
		log_manager.info(ret2)
		log_manager.info("")
		"""
		ret3 = model.random_generate_session(sess)
		log_manager.info(ret3)
		log_manager.info("")		
		"""


if __name__ == '__main__':
	tf.app.run()