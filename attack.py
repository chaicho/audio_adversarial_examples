# attack.py -- generate audio adversarial examples
##
# Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
# This program is licenced under the BSD 2-Clause licence,
# contained in the LICENCE file in this directory.

import numpy as np
from random import randint
from numpy.core.fromnumeric import shape
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from tensorflow.keras.backend import ctc_label_dense_to_sparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tf_logits import get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"


class Attack:
    def __init__(self, max_audio_length, max_target_phrase_length,
                 batch_size=1,
                 learning_rate=10,
                 l2penalty=float('inf')):
        # Basic information for building inference graph
        self.batch_size = batch_size
        self.max_target_phrase_length = max_target_phrase_length
        self.max_audio_length = max_audio_length

        # Trainable variables
        self.delta = tf.Variable(tf.zeros(2 * max_audio_length, dtype=tf.float32),
                                 name='qq_delta')
        # Placeholder
        self.mask = tfv1.placeholder(shape=(batch_size, max_audio_length),
                                   dtype=tf.bool)
        self.audio = tfv1.placeholder(shape=(batch_size, max_audio_length),
                                    dtype=tf.float32)
        self.length = tfv1.placeholder(shape=(batch_size,),
                                     dtype=tf.int32)
        self.target_phrase = tfv1.placeholder(shape=(batch_size, max_target_phrase_length),
                                            dtype=tf.int32)
        self.target_phrase_length = tfv1.placeholder(shape=(batch_size,),
                                                   dtype=tf.int32)
        self.interval_start = tf.placeholder( 
                                        dtype =tf.int32)
        self.rescale = tf.Variable(tf.ones((1,), dtype=tf.float32),
                                   name='qq_delta')
        self.is_command = tf.placeholder( 
                                        dtype =tf.int32)
        # Prepare input audios
        apply_delta = tf.clip_by_value((self.delta)[self.interval_start: self.interval_start + max_audio_length], -2000, 2000)
        apply_delta = apply_delta * self.rescale * \
            tf.cast(self.mask, tf.float32)
        noise = tf.random.normal((batch_size, max_audio_length), stddev=2)
        self.noised_audio = tf.clip_by_value(
            self.audio + apply_delta + noise, -2**15, 2**15-1)

        # Get inference result of DeepSpeech
        self.logits = get_logits(self.noised_audio, self.length)
        self.decoded, _ = tfv1.nn.ctc_beam_search_decoder(
            self.logits, self.length, merge_repeated=False, beam_width=100)

        # Calculate loss
        target = ctc_label_dense_to_sparse(
            self.target_phrase, self.target_phrase_length)
        self.ctc_loss = tfv1.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                      inputs=self.logits, sequence_length=self.length)
        if not np.isinf(l2penalty):
            l2diff = tf.reduce_mean(
                (self.noised_audio-self.audio)**2, axis=1)
            loss = l2diff + l2penalty*self.ctc_loss
        else:
            loss = self.ctc_loss
        self.loss = loss 

        # Optimize step
        self.lr = tf.Variable(0.0, shape=(), name='qq_lr')
        global_step = tfv1.train.get_global_step()
        # self.lr = tfv1.train.exponential_decay(
        #     1e-6, global_step=global_step,
        #     decay_steps=10, decay_rate=2)
        self.optimizer = tfv1.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.loss * self.is_command, global_step=global_step, var_list=[self.delta])
        tf.summary.scalar("learning_rate", self.lr)
        tf.summary.scalar("current_step", global_step)
        tf.summary.scalar("loss", self.loss)
        # self.optimizer = tfv1.train.AdamOptimizer(learning_rate)
        # grad, var = self.optimizer.compute_gradients(
        #     self.loss, [self.delta])[0]
        # self.train_op = self.optimizer.apply_gradients([(-grad, var)])

    def init_sess(self, sess, restore_path=None):
        # And finally restore the graph to make the classifier
        # actually do something interesting.
        saver = tf.train.Saver(
            [x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)

        sess.run(tf.variables_initializer(
            self.optimizer.variables() + [self.delta, self.rescale]))

    def build_feed_dict(self, audios, lengths, target=None,is_command = -1):
        assert audios.shape[0] == self.batch_size
        assert audios.shape[1] == self.max_audio_length
        assert lengths.shape[0] == self.batch_size

        masks = np.zeros(
            (lengths.shape[0], self.max_audio_length), dtype=np.bool)
        for i, l in enumerate(lengths):
            masks[i, :l] = True

        feed_dict = {
            self.audio: audios,
            self.length: (lengths-1)//320,
            self.mask: masks,
            self.interval_start : randint(0,self.max_audio_length),
            self.is_command :  is_command
        }
        # print(self.interval_start)
        if target is not None:
            target = [[toks.index(x) for x in phrase.lower()] for phrase in target]
            feed_dict = {
                **feed_dict,
                self.target_phrase: pad_sequences(target,
                                                  maxlen=self.max_target_phrase_length),
                self.target_phrase_length: [len(x) for x in target],

            }

        return feed_dict

    def inference(self, sess, feed_dict):
        out, logits = sess.run((self.decoded, self.logits),
                               feed_dict=feed_dict)

        res = np.zeros(out[0].dense_shape) + len(toks) - 1
        for ii in range(len(out[0].values)):
            x, y = out[0].indices[ii]
            res[x, y] = out[0].values[ii]

        # the strings that are recognized.
        res = ["".join(toks[int(x)]
                       for x in y).replace("-", "") for y in res]

        # the argmax of the alignment.
        res2 = ["".join(toks[int(x)] for x in y[:l])
                for y, l in zip(np.argmax(logits, axis=2).T, feed_dict[self.length])]
        return res, res2

    def train_step(self, sess, feed_dict):
        loss, _ = sess.run((self.loss, self.train_op),
                            feed_dict=feed_dict)
        return loss

    def get_delta(self, sess):
        delta = sess.run(tf.clip_by_value(model.delta, -2000, 2000) * self.rescale)

    def train(self, sess, target, feed_dict, iterations=100):
        target_tokens = ["".join([toks[x] for x in t]) for t in target]
        
        for i in range(iterations):
            # Print debug informations
            if i % 10 == 0:
                res, _ = self.inference(sess, feed_dict=feed_dict)

                worked = [ii for ii in range(
                    self.batch_size) if res[ii] == target_tokens[ii]]
                if len(worked) > 0:
                    print('Worked for', *worked)
                if len(worked) == self.batch_size:
                    delta, rescale = sess.run((self.delta, self.rescale))
                    delta.numpy().save('delta.npy')
                    rescale.numpy().save('rescale.npy')
                    sess.run(self.rescale.assign(self.rescale * 0.8))
                    print('Work to all audios, threshold changed to {}'.format(
                        rescale * 0.8))

            loss = self.train_step(sess, feed_dict)

            # Report progress
            print("step {}: loss=({:.3f} {:.3f} {:.3f})".format(
                i, loss.mean(), loss.min(), loss.max()))
