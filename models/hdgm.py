# -*- coding:utf-8 -*-

""" Tensorflow Implementation of two-level Hierarchical Deep Generative Models with Dual Memory.
	Paper: Sequence Modeling with Hierarchical Deep Generative Models with Dual Memory
	(https://dl.acm.org/citation.cfm?id=3132952)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class HDGM(object):
	def __init__(self, hparam, word_to_index, index_to_word):
		self.hparam = hparam
		self.word_to_index = word_to_index
		self.index_to_word = index_to_word

		self.seq_len = hparam.seq_len
		self.vocab_size = hparam.vocab_size
		self.embedding_dim = hparam.embedding_dim
		self.hidden_dim = hparam.hidden_dim
		self.latent_dim = hparam.latent_dim
		self.clip_norm = hparam.clip_norm
		self.num_layers = hparam.num_layers
		self.beam_width = hparam.beam_width
		self.rnn_type = hparam.rnn_type
		self.anneal_max = hparam.anneal_max
		self.anneal_bias = hparam.anneal_bias
		self.anneal_slope = hparam.anneal_slope

		self.bidirect = hparam.bidirect
		self.attention_dim = hparam.attention_dim

		self.SOS_ID = word_to_index["<sos>"]
		self.EOS_ID = word_to_index["<eos>"]
		self.UNK_ID = word_to_index["<unk>"]
		self.PAD_ID = word_to_index["<pad>"]

		self._build_global_helper()
		self._build_graph()
		self._build_summary()

		self.merged_summary = tf.summary.merge_all()
		self.init = tf.global_variables_initializer()

	def _build_global_helper(self):
		self.inputs = tf.placeholder(tf.int32, [None, self.seq_len])
		self.targets = tf.placeholder(tf.int32, [None, self.seq_len])

		self.enc_inputs = self.inputs
		self.dec_inputs = self.inputs
		self.dec_targets = self.targets
		self.enc_lengths = tf.count_nonzero(self.enc_inputs, 1, dtype=tf.int32)
		self.dec_lengths = self.enc_lengths
		self.batch_size = tf.shape(self.inputs)[0]

		self.lr = tf.Variable(0.0, trainable=False)
		self.new_lr = tf.placeholder(tf.float32, [])
		self.lr_update = tf.assign(self.lr, self.new_lr)

		self.global_step = tf.Variable(0, trainable=False)
		self.saver = tf.train.Saver()

	def _build_graph(self):
		self._build_forward_graph()
		self._build_backward_graph()



	### MLP attention
	def attention(self, rnn_outputs):
			if isinstance(rnn_outputs, tuple):
				inputs = tf.concat(rnn_outputs, 2) # [B,T,2H]
				hidden_dim = self.hidden_dim * 2
			else:
				inputs = rnn_outputs # [B,T,H]
				hidden_dim = self.hidden_dim
			
			w_omega = tf.Variable(tf.random_normal([hidden_dim, self.attention_dim], stddev=0.1))
			b_omega = tf.Variable(tf.random_normal([self.attention_dim], stddev=0.1))
			u_omega = tf.Variable(tf.random_normal([self.attention_dim], stddev=0.1)) # [A]

			new_inputs = tf.reshape(inputs, [-1, hidden_dim])
			v = tf.tanh(tf.matmul(new_inputs, w_omega) + b_omega) # [B*T, A]

			vu = tf.reduce_sum(tf.multiply(v, u_omega), -1) # [B*T, A]
			vu = tf.reshape(vu, [-1, self.seq_len])

			# v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
			# vu = tf.tensordot(v, u_omega, axes=1, name='vu')

			alphas = tf.nn.softmax(vu, name='alphas') # [B*T, A]
			outputs = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1) # [batch_size, attention]
			# output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
			# [batch_size, dim]
			return outputs, alphas


	def inference_network(self, broad_memory, last_state):

		inputs = tf.concat([broad_memory, last_state], -1) # [batch_size, hidden_dim]
		mean = tf.layers.dense(inputs, self.latent_dim, name="enc_mean")
		logvar = tf.layers.dense(inputs, self.latent_dim, name="enc_logvar")
		epsilon = tf.truncated_normal(tf.shape(mean))
		latent_sample = mean + tf.exp(0.5 * logvar) * epsilon

		return latent_sample, mean, logvar




	def _build_forward_graph(self):

		# encoder
		with tf.variable_scope("encoder", reuse=None):
			embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_dim], dtype=tf.float32)
			self.embedded_enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_inputs)
			self.embedded_dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)

			if self.bidirect == True:
				fw_cell = self.create_rnn_cell(reuse=None)
				bw_cell = self.create_rnn_cell(reuse=None)

				enc_outputs, enc_states = tf.nn.bidirectional_dynamic_rnn(
					cell_fw=fw_cell,
					cell_bw=bw_cell,
					inputs=self.embedded_enc_inputs,
					sequence_length=self.enc_lengths,
					dtype = tf.float32)
			else:
				enc_cell = self.create_rnn_cell(reuse=None)
				enc_init = enc_cell.zero_state(self.batch_size, tf.float32)
				enc_outputs, enc_states = tf.nn.dynamic_rnn(cell = enc_cell,
														initial_state = enc_init,
														inputs = self.embedded_enc_inputs,
														sequence_length = self.enc_lengths,
														dtype = tf.float32)
				# [batch_size, seq_len, hidden_dim]

			self.broad_memory, self.alphas = self.attention(enc_outputs)
			# batch_size, dim
		
		# inference network
		with tf.variable_scope("inference_network_1", reuse=None):
			self.latent_1, self.mean_1, self.logvar_1 = self.inference_network(self.broad_memory, self.broad_memory)

		with tf.variable_scope("inference_network_2", reuse=None):
			self.latent_2, self.mean_2, self.logvar_2 = self.inference_network(self.broad_memory, self.latent_1)


		with tf.variable_scope("generative_network_2", reuse=None):
			inputs = tf.concat([self.broad_memory, self.latent_2], -1)
			self.gen_mean = tf.layers.dense(inputs, self.latent_dim, name="gen_mean")
			self.gen_logvar = tf.layers.dense(inputs, self.latent_dim, name="gen_logvar")
			gen_epsilon = tf.truncated_normal(tf.shape(self.gen_mean))
			self.gen_latent_sample = self.gen_mean + tf.exp(0.5 * self.gen_logvar) * gen_epsilon


		gen_inputs = tf.concat([self.broad_memory, self.latent_1, self.latent_2, self.gen_latent_sample], -1)
		self.training_rnn_outputs, self.training_logits = self.decoder_training(gen_inputs)
		self.reconstructions = tf.argmax(self.training_logits, 2)
		self.predicted_ids = self.decoder_inference(gen_inputs)


	def _build_backward_graph(self):
		self.nll_loss = self.nll_loss_fn()
		self.kl_w = self.kl_w_fn()

		self.kl_loss_1 = self.kl_loss_fn()
		self.kl_loss_2 = self.kl_loss_between_fn(self.mean_2, self.logvar_2, self.gen_mean, self.gen_logvar)
		self.kl_loss = self.kl_loss_1 + self.kl_loss_2

		self.total_loss = self.nll_loss + self.kl_loss * self.kl_w
		self.elbo = self.total_loss

		params = tf.trainable_variables()
		gradients = tf.gradients(self.total_loss, params)
		clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
		self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(
			zip(clipped_gradients, params), global_step=self.global_step)

	def create_rnn_cell(self, reuse):
		def create_single_cell(reuse):
			if self.rnn_type == "gru":
				cell = tf.contrib.rnn.GRUCell(self.hidden_dim, 
										  	  kernel_initializer=tf.orthogonal_initializer(),
										  	  reuse=reuse)
			elif self.rnn_type == "block":
				cell = tf.contrib.rnn.LSTMBlockCell(self.hidden_dim, forget_bias=0.0, reuse=reuse)
			elif self.rnn_type == "basic":
				cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=0.0, state_is_tuple=True, reuse=reuse)
			else:
				ValueError("Unsupported rnn_type.")
			return cell
		cell = tf.contrib.rnn.MultiRNNCell([create_single_cell(reuse) for _ in range(self.num_layers)], state_is_tuple=True)
		return cell


	def _build_summary(self):
		tf.summary.scalar("total_loss", self.total_loss)
		tf.summary.scalar("nll_loss", self.nll_loss)
		tf.summary.scalar("kl_loss", self.kl_loss)
		tf.summary.scalar("kl_w", self.kl_w)
		# tf.summary.scalar("ratio", self.nll_loss / self.kl_loss)


	def decoder_training(self, latent_sample):
		with tf.variable_scope("encoder", reuse=True):
			embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_dim], dtype=tf.float32)
		with tf.variable_scope("decoder", reuse=None):
			if self.rnn_type == "gru":
				init_state = tuple([tf.layers.dense(latent_sample, self.hidden_dim, tf.nn.elu, name="trans0")] * self.num_layers)
			else:
				c = tf.layers.dense(latent_sample, self.hidden_dim, tf.nn.elu, name="trans1")
				h = tf.layers.dense(latent_sample, self.hidden_dim, tf.nn.elu, name="trans2")
				init_state = tuple([tf.contrib.rnn.LSTMStateTuple(c=c, h=h)] * self.num_layers)

			lin_proj = tf.layers.Dense(self.vocab_size, _scope='decoder/dense', _reuse=None)
			dec_cell = self.create_rnn_cell(reuse=None)
			helper = tf.contrib.seq2seq.TrainingHelper(inputs = self.embedded_dec_inputs,
													   sequence_length = self.dec_lengths)
			decoder = tf.contrib.seq2seq.BasicDecoder(cell = dec_cell,
													  helper = helper,
													  initial_state = init_state)
			decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = decoder)
		return decoder_output.rnn_output, lin_proj.apply(decoder_output.rnn_output)


	def decoder_inference(self, latent_sample):
		with tf.variable_scope("encoder", reuse=True):
			embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_dim], dtype=tf.float32)
		with tf.variable_scope("decoder", reuse=True):
			if self.rnn_type == "gru":
				init_state = tuple([tf.layers.dense(latent_sample, self.hidden_dim, tf.nn.elu, name="trans0")] * self.num_layers)
			else:
				c = tf.layers.dense(latent_sample, self.hidden_dim, tf.nn.elu, name="trans1")
				h = tf.layers.dense(latent_sample, self.hidden_dim, tf.nn.elu, name="trans2")
				init_state = tuple([tf.contrib.rnn.LSTMStateTuple(c=c, h=h)] * self.num_layers)
			
			lin_proj = tf.layers.Dense(self.vocab_size, _scope='decoder/dense', _reuse=True)
			dec_cell = self.create_rnn_cell(reuse=True)
			decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=dec_cell,
													   embedding=embedding,
													   start_tokens=tf.tile(tf.constant([self.SOS_ID], dtype=tf.int32), [self.batch_size]),
													   end_token = tf.constant(self.EOS_ID, dtype=tf.int32),
													   initial_state = tf.contrib.seq2seq.tile_batch(init_state, self.beam_width),
													   beam_width = self.beam_width,
													   output_layer = lin_proj)
			decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = decoder,
											maximum_iterations = 2 * tf.reduce_max(self.enc_lengths))
		return decoder_output.predicted_ids[:, :, 0]


	def nll_loss_fn(self):
		mask = tf.sequence_mask(self.dec_lengths, tf.reduce_max(self.dec_lengths), dtype=tf.float32)
		return tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
					logits = self.training_logits,
					targets = self.dec_targets,
					weights = mask,
					average_across_timesteps = False,
					average_across_batch = True))
	

	def kl_w_fn(self):
		return self.anneal_max * tf.sigmoid((self.anneal_slope) * (tf.cast(self.global_step, tf.float32) - tf.constant(self.anneal_bias / 2)))


	def kl_loss_fn(self):
		return 0.5 * tf.reduce_sum(tf.exp(self.logvar_1) + tf.square(self.mean_1) - 1 - self.logvar_1) / tf.to_float(self.batch_size)

	def kl_loss_between_fn(self, mu1, logvar1, mu2, logvar2):
		return -0.5 * tf.reduce_sum(1 + logvar1 - logvar2 - tf.exp(logvar1) / tf.exp(logvar2) -
					   ((mu1 - mu2)**2) / tf.exp(logvar2)) / tf.to_float(self.batch_size)



	def train_session(self, sess, inputs, targets):
		feed_dict = {self.inputs: inputs,
					 self.targets: targets}
		fetches = [self.train_op,
				   self.nll_loss,
				   self.kl_loss,
				   self.kl_w,
				   self.total_loss,
				   self.global_step,
				   self.merged_summary]
		_, nll_loss, kl_loss, kl_w, total_loss, step, summary = sess.run(fetches, feed_dict)
		return {"nll_loss": nll_loss,
				"kl_loss": kl_loss,
				"kl_w": kl_w,
				"total_loss": total_loss,
				"step": step,
				"summary": summary}


	def test_session(self, sess, inputs, targets):
		feed_dict = {self.inputs: inputs,
					 self.targets: targets}
		nll_loss, total_loss, kl_loss = sess.run([self.nll_loss, self.total_loss, self.kl_loss], feed_dict)
		return {"nll_loss": nll_loss, 
				"total_loss": total_loss,
				"kl_loss": kl_loss}


	"""
	def random_generate_session(self, sess):
		feed_dict = {self.latent_sample: np.random.randn(1, self.latent_dim),
					 self.enc_lengths: [self.seq_len],
					 self.batch_size: 1}
		predicted_ids = sess.run(self.predicted_ids, feed_dict)[0]
		ret = 'Random Generation: %s' % ' '.join([self.index_to_word[index] for index in predicted_ids])
		return ret
	"""

	def reconstruct_session(self, sess, sentence):
		idx_list = [self.get_word_index(w) for w in sentence.lower().split()][:self.seq_len]
		idx_list = idx_list + [self.PAD_ID] * (self.seq_len - len(idx_list))
		ret1 = 'Original: %s' % ' '.join([self.index_to_word[index] for index in idx_list])
		
		predicted_ids = sess.run(self.reconstructions, {self.inputs: np.atleast_2d(idx_list)})[0]
		ret2 = 'Reconstr: %s' % ' '.join([self.index_to_word[index] for index in predicted_ids])
		return ret1, ret2


	def get_word_index(self, word):
		index = self.word_to_index[word]
		return index if index < self.vocab_size else self.UNK_ID


	def get_parameter_size(self):
		all_vars = tf.global_variables()
		total_count = 0
		for item in all_vars:
			if "Adam" in item.name:
				continue
			shape = item.get_shape().as_list()
			if len(shape) == 0:
				total_count += 1
			else:
				size =  1
				for val in shape:
					size *= val
				total_count += size
		return total_count