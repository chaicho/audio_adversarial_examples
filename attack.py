# attack.py -- generate audio adversarial examples
##
# Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
# This program is licenced under the BSD 2-Clause licence,
# contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from tensorflow.keras.backend import ctc_label_dense_to_sparse
from tensorflow.compat.v1.nn import ctc_beam_search_decoder
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
        self.delta = tf.Variable(
            tf.zeros(max_audio_length, dtype=tf.float32), name='qq_delta')

        # Placeholder
        self.mask = tf.Variable(
            tf.zeros((batch_size, max_audio_length), dtype=tf.float32), name='qq_mask')
        self.audios = tf.Variable(
            tf.zeros((batch_size, max_audio_length), dtype=tf.float32), name='qq_original')
        self.lengths = tf.Variable(
            tf.zeros(batch_size, dtype=tf.int32), name='qq_lengths')
        self.importance = tf.Variable(tf.zeros(
            (batch_size, max_target_phrase_length), dtype=tf.float32), name='qq_importance')
        self.target_phrase = tf.Variable(tf.zeros(
            (batch_size, max_target_phrase_length), dtype=tf.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(
            tf.zeros((batch_size), dtype=tf.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(
            tf.zeros((1,), dtype=tf.float32), name='qq_phrase_lengths')

        # Prepare input audios
        apply_delta = tf.clip_by_value(
            self.delta, -2000, 2000)*self.rescale*self.mask
        noise = tf.random.normal((batch_size, max_audio_length), stddev=2)
        self.noised_audios = tf.clip_by_value(
            self.audios + apply_delta + noise, -2**15, 2**15-1)

        # Get inference result of DeepSpeech
        self.logits = get_logits(self.noised_audios, self.lengths)
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(
            self.logits, self.lengths, merge_repeated=False, beam_width=100)

        # Calculate loss
        target = ctc_label_dense_to_sparse(
            self.target_phrase, self.target_phrase_lengths)
        self.ctc_loss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                       inputs=self.logits, sequence_length=self.lengths)
        if not np.isinf(l2penalty):
            l2diff = tf.reduce_mean(
                (self.noised_audios-self.audios)**2, axis=1)
            loss = l2diff + l2penalty*self.ctc_loss
        else:
            loss = self.ctc_loss
        self.loss = loss

        # Optimize step
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        grad, var = self.optimizer.compute_gradients(
            self.loss, [self.delta])[0]
        self.train_op = self.optimizer.apply_gradients([(tf.sign(grad), var)])

    def init_sess(self, sess, restore_path=None):
        # And finally restore the graph to make the classifier
        # actually do something interesting.
        saver = tf.train.Saver(
            [x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)

        sess.run(tf.variables_initializer(
            self.optimizer.variables()+[self.delta]))

    def inference(self, sess, audios, lengths):
        sess.run(self.audios.assign(audios))
        sess.run(self.lengths.assign((lengths-1)//320))
        sess.run(self.mask.assign(
            [[1 if i < l else 0 for i in range(self.max_audio_length)] for l in lengths]))
        out, logits = sess.run((self.decoded, self.logits))

        res = np.zeros(out[0].dense_shape) + len(toks) - 1
        for ii in range(len(out[0].values)):
            x, y = out[0].indices[ii]
            res[x, y] = out[0].values[ii]

        # Here we print the strings that are recognized.
        res = ["".join(toks[int(x)]
                       for x in y).replace("-", "") for y in res]
        print("\n".join(res))

        # And here we print the argmax of the alignment.
        res2 = np.argmax(logits, axis=2).T
        res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320])
                for y, l in zip(res2, lengths)]
        print("\n".join(res2))

    def train(self, sess, audios, lengths, targets, iterations=100):
        sess.run(self.audios.assign(audios))
        sess.run(self.lengths.assign((lengths-1)//320))
        sess.run(self.mask.assign(
            [[1 if i < l else 0 for i in range(self.max_audio_length)] for l in lengths]))
        sess.run(self.target_phrase_lengths.assign([len(x) for x in targets]))
        sess.run(self.target_phrase.assign(
            [list(t)+[0]*(self.max_target_phrase_length-len(t)) for t in targets]))
        sess.run(self.rescale.assign(tf.ones((1,))))

        target_tokens = ["".join([toks[x] for x in target])
                         for target in targets]

        for i in range(iterations):
            # Print out some debug information every 10 iterations.
            if i % 10 == 0:
                out, logits = sess.run((self.decoded, self.logits))
                res = np.zeros(out[0].dense_shape)+len(toks)-1

                for ii in range(len(out[0].values)):
                    x, y = out[0].indices[ii]
                    res[x, y] = out[0].values[ii]

                # Here we print the strings that are recognized.
                res = ["".join(toks[int(x)]
                               for x in y).replace("-", "") for y in res]
                print("\n".join(res))

                # And here we print the argmax of the alignment.
                res2 = np.argmax(logits, axis=2).T
                res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320])
                        for y, l in zip(res2, lengths)]
                print("\n".join(res2))

                worked = [ii for ii in range(
                    self.batch_size) if res[ii] == target_tokens[ii]]
                if len(worked) > 0:
                    print('Worked for', *worked)
                if len(worked) == self.batch_size:
                    delta, rescale = sess.run((self.delta, self.rescale))
                    delta.numpy().save('delta.npy')
                    rescale.numpy().save('rescale.npy')
                    sess.run(self.rescale.assign(self.rescale * 0.8))
                    print('Work to all audios, threshold changed to {}'.format(rescale * 0.8))

            # Actually do the optimization step
            ctc_loss, _ = sess.run((self.ctc_loss, self.train_op))

            # Report progress
            print("step {}: {:.3f}{}".format(i, np.mean(ctc_loss),
                  "\t".join("{:8.3f}".format(x) for x in ctc_loss)))
