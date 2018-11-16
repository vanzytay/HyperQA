#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division

import os
import pickle
import timeit
from typing import Tuple, List, Dict

from tqdm import tqdm
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score

from datasets.Base import BaseQA
from datasets.wikiqa import WikiQA
from glove import load_embedding_from_disks
from parser import build_parser
from utilities import *
from datasets.yahooqa import YahooQA

tf.logging.set_verbosity(tf.logging.INFO)


def batchify(data, i, bsz, max_sample):
    start = int(i * bsz)
    end = int(i * bsz) + bsz
    if max_sample < end:
        end = max_sample
    data = tuple(t[start:end] for t in data)
    return data


def get_ds(name, path, word_to_index, index_to_embedding, qmax, amax, char_min, num_neg):
    if name == 'yahooqa':
        dataset = YahooQA(path, word_to_index, index_to_embedding, qmax, amax, char_min, num_neg)
        tf.logging.info('YahooDS loaded')
    elif name == 'wikiqa':
        dataset = WikiQA(path, word_to_index, index_to_embedding, qmax, amax, char_min, num_neg)
        tf.logging.info('WikiQA loaded')
    return dataset


class HyperQA:
    """
    Hyperbolic Embeddings for QA

    """

    def __init__(self, dataset: BaseQA, char_vocab=0, pos_vocab=0):
        tf.set_random_seed(4242)
        self.parser = build_parser()
        self.char_vocab = char_vocab
        self.pos_vocab = pos_vocab
        self.graph = tf.Graph()
        self.args = self.parser.parse_args()
        self.imap = {}
        self.inspect_op = []
        self.feat_prop = None

        self._train_set = None
        self._test_set = None
        self._dev_set = None

        self.word_to_index, self.index_to_embedding = dataset.word_to_index, dataset.index_to_embedding
        self.index_to_word = {val: k for k, val in self.word_to_index.items()}
        self.vocab_size = len(self.index_to_embedding)

        if self.args.init_type == 'xavier':
            self.initializer = tf.contrib.layers.xavier_initializer()
        elif self.args.init_type == 'normal':
            self.initializer = tf.random_normal_initializer()
        elif self.args.init_type == 'uniform':
            self.initializer = tf.random_uniform_initializer(maxval=self.args.init, minval=-self.args.init)
        else:
            self.initializer = tf.contrib.layers.xavier_initializer()

        self.build_graph()
        _config_proto = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=8)
        self.sess = tf.Session(graph=self.graph, config=_config_proto)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

        self.ckpt_path = os.path.join(args.ckpt_path, self.__class__.__name__)

    @property
    def train_set(self):
        if self._train_set is None:
            tf.logging.info('Create train split')
            self._train_set = dataset.splits[dataset.Parts.train.name]
        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            tf.logging.info('Create test split')
            self._test_set = dataset.splits[dataset.Parts.test.name]
        return self._test_set

    @property
    def dev_set(self):
        if self._dev_set is None:
            tf.logging.info('Create dev split')
            self._dev_set = dataset.splits[dataset.Parts.dev.name]
        return self._dev_set

    def _get_pair_feed_dict(self, data, mode='training', lr=None):

        if lr is None:
            lr = self.args.learn_rate

        if mode == 'training':
            assert (np.min(data[1]) > 0)
            assert (np.min(data[3]) > 0)
            assert (np.min(data[5]) > 0)

        feed_dict = {
            # question
            self.q1_inputs: data[0],
            self.q1_len: data[1],
            # positive answer
            self.q2_inputs: data[2],
            self.q2_len: data[3],
            # negative answer
            self.q3_inputs: data[4],
            self.q3_len: data[5],
            self.learn_rate: lr,
            self.dropout: self.args.dropout,
            self.emb_dropout: self.args.emb_dropout
        }

        if mode != 'training':
            feed_dict[self.dropout] = 1.0
            feed_dict[self.emb_dropout] = 1.0

        return feed_dict

    def get_feed_dict(self, data, mode='training', lr=None):
        return self._get_pair_feed_dict(data, mode=mode, lr=lr)

    def register_index_map(self, idx, target):
        self.imap[target] = idx

    def build_glove(self, embed, lens, max_len):
        embed = mask_zeros_1(embed, lens, max_len)
        return tf.reduce_sum(embed, 1)

    def learn_repr(self, q1_embed, q2_embed, q1_len, q2_len, q1_max,
                   q2_max, force_model=None, score=1,
                   reuse=None, extract_embed=False,
                   side=''):

        translate_act = tf.nn.relu
        use_mode = 'FC'

        q1_embed = projection_layer(
            q1_embed,
            self.args.rnn_size,
            name='trans_proj',
            activation=translate_act,
            initializer=self.initializer,
            dropout=self.dropout,
            reuse=reuse,
            use_mode=use_mode,
            num_layers=self.args.num_proj
        )
        q2_embed = projection_layer(
            q2_embed,
            self.args.rnn_size,
            name='trans_proj',
            activation=translate_act,
            initializer=self.initializer,
            dropout=self.dropout,
            reuse=True,
            use_mode=use_mode,
            num_layers=self.args.num_proj
        )

        rnn_size = self.args.rnn_size

        q1_output = self.build_glove(q1_embed, q1_len, q1_max)
        q2_output = self.build_glove(q2_embed, q2_len, q2_max)

        try:
            self.max_norm = tf.reduce_max(tf.norm(q1_output, ord='euclidean', keepdims=True, axis=1))
        except:
            self.max_norm = 0

        if extract_embed:
            self.q1_extract = q1_output
            self.q2_extract = q2_output

        q1_output = tf.nn.dropout(q1_output, self.dropout)
        q2_output = tf.nn.dropout(q2_output, self.dropout)

        # This constraint is important
        _q1_output = tf.clip_by_norm(q1_output, 1.0, axes=1)
        _q2_output = tf.clip_by_norm(q2_output, 1.0, axes=1)
        output = hyperbolic_ball(_q1_output, _q2_output)

        representation = output
        activation = None

        with tf.variable_scope('fl', reuse=reuse) as scope:
            last_dim = output.get_shape().as_list()[1]
            num_outputs = 1
            weights_linear = tf.get_variable('final_weights', [last_dim, num_outputs], initializer=self.initializer)
            bias_linear = tf.get_variable('bias', [num_outputs], initializer=tf.zeros_initializer())
            final_layer = tf.nn.xw_plus_b(output, weights_linear, bias_linear)
            output = final_layer

        return output, representation

    def build_graph(self):
        ''' Builds Computational Graph
        '''

        with self.graph.as_default():
            with tf.name_scope('q1_input'):
                self.q1_inputs = tf.placeholder(tf.int32, shape=[None, None], name='q1_inputs')

            with tf.name_scope('q2_input'):
                self.q2_inputs = tf.placeholder(tf.int32, shape=[None, None], name='q2_inputs')

            with tf.name_scope('q3_input'):
                self.q3_inputs = tf.placeholder(tf.int32, shape=[None, None], name='q3_inputs')

            with tf.name_scope('dropout'):
                self.dropout = tf.placeholder(tf.float32, name='dropout')
                self.emb_dropout = tf.placeholder(tf.float32, name='emb_dropout')

            with tf.name_scope('q1_lengths'):
                self.q1_len = tf.placeholder(tf.int32, shape=[None])

            with tf.name_scope('q2_lengths'):
                self.q2_len = tf.placeholder(tf.int32, shape=[None])

            with tf.name_scope('q3_lengths'):
                self.q3_len = tf.placeholder(tf.int32, shape=[None])

            with tf.name_scope('learn_rate'):
                self.learn_rate = tf.placeholder(tf.float32, name='learn_rate')

            if self.args.pretrained == 1:
                self.emb_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.args.emb_size])

            self.batch_size = tf.shape(self.q1_inputs)[0]

            q1_inputs, self.qmax = clip_sentence(self.q1_inputs, self.q1_len)
            q2_inputs, self.a1max = clip_sentence(self.q2_inputs, self.q2_len)
            q3_inputs, self.a2max = clip_sentence(self.q3_inputs, self.q3_len)

            with tf.variable_scope('embedding_layer'):
                if self.args.pretrained == 1:
                    self.embeddings = tf.Variable(
                        tf.constant(0.0, shape=[self.vocab_size, self.args.emb_size]),
                        # trainable=self.args.trainable,
                        trainable=False,
                        name="embeddings"
                    )
                    self.embeddings_init = self.embeddings.assign(self.emb_placeholder)
                else:
                    self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.args.emb_size], -0.01, 0.01))

                q1_embed = tf.nn.embedding_lookup(self.embeddings, q1_inputs)
                q2_embed = tf.nn.embedding_lookup(self.embeddings, q2_inputs)
                q3_embed = tf.nn.embedding_lookup(self.embeddings, q3_inputs)

            if self.args.all_dropout:
                q1_embed = tf.nn.dropout(q1_embed, self.emb_dropout)
                q2_embed = tf.nn.dropout(q2_embed, self.emb_dropout)
                q3_embed = tf.nn.dropout(q3_embed, self.emb_dropout)

            compose_length = self.args.rnn_size
            rnn_length = self.args.rnn_size
            repr_fun = self.learn_repr

            self.output_pos, _ = repr_fun(q1_embed, q2_embed, self.q1_len, self.q2_len, self.qmax, self.a1max, score=1,
                                          reuse=None, extract_embed=True, side='POS')

            self.output_neg, _ = repr_fun(q1_embed, q3_embed, self.q1_len, self.q3_len, self.qmax, self.a2max, score=1,
                                          reuse=True, side='NEG')

            # Define loss and optimizer
            with tf.name_scope("train"):
                with tf.name_scope("cost_function"):
                    # hinge loss
                    self.hinge_loss = tf.maximum(0.0, (self.args.margin - self.output_pos + self.output_neg))

                    self.cost = tf.reduce_sum(self.hinge_loss)

                    with tf.name_scope('regularization'):
                        if self.args.l2_reg > 0:
                            vars = tf.trainable_variables()
                            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])
                            lossL2 *= self.args.l2_reg
                            self.cost += lossL2

                    tf.summary.scalar("cost_function", self.cost)
                global_step = tf.Variable(0, trainable=False)

                if self.args.decay_lr > 0 and self.args.decay_epoch > 0:
                    decay_epoch = self.args.decay_epoch
                    lr = tf.train.exponential_decay(self.args.learn_rate,
                                                    global_step,
                                                    decay_epoch * self.args.batch_size,
                                                    self.args.decay_lr, staircase=True)
                else:
                    lr = self.args.learn_rate

                with tf.name_scope('optimizer'):
                    if self.args.opt == 'SGD':
                        self.opt = tf.train.GradientDescentOptimizer(
                            learning_rate=lr)
                    elif self.args.opt == 'Adam':
                        self.opt = tf.train.AdamOptimizer(
                            learning_rate=lr)
                    elif self.args.opt == 'Adadelta':
                        self.opt = tf.train.AdadeltaOptimizer(
                            learning_rate=lr,
                            rho=0.9)
                    elif self.args.opt == 'Adagrad':
                        self.opt = tf.train.AdagradOptimizer(
                            learning_rate=lr)
                    elif self.args.opt == 'RMS':
                        self.opt = tf.train.RMSPropOptimizer(
                            learning_rate=lr)
                    elif self.args.opt == 'Moment':
                        self.opt = tf.train.MomentumOptimizer(lr, 0.9)
                    # elif (self.args.opt == 'Adamax'):
                    #     self.opt = AdamaxOptimizer(lr)

                    # Use SGD at the end for better local minima
                    tvars = tf.trainable_variables()

                    def _none_to_zero(grads, var_list):
                        return [grad if grad is not None else tf.zeros_like(var)
                                for var, grad in zip(var_list, grads)]

                    if self.args.clip_norm > 0:
                        grads, _ = tf.clip_by_global_norm(
                            tf.gradients(self.cost, tvars),
                            self.args.clip_norm)
                        with tf.name_scope('gradients'):
                            gradients = self.opt.compute_gradients(self.cost)
                            # Gradient Conversion
                            gradients = [(H2E_ball(grad), var) for grad, var in gradients]

                            def ClipIfNotNone(grad):
                                if grad is None:
                                    return grad
                                grad = tf.clip_by_value(grad, -10, 10, name=None)
                                return tf.clip_by_norm(grad, self.args.clip_norm)

                            if self.args.clip_norm > 0:
                                clip_g = [(ClipIfNotNone(grad), var) for grad, var in gradients]
                            else:
                                clip_g = [(grad, var) for grad, var in gradients]

                        control_deps = None
                        with tf.control_dependencies(control_deps):
                            self.train_op = self.opt.apply_gradients(clip_g, global_step=global_step)
                            # self.wiggle_op = self.opt2.apply_gradients(clip_g,
                            #                                            global_step=global_step)

                self.grads = _none_to_zero(tf.gradients(self.cost, tvars), tvars)
                self.merged_summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
                self.predict_op = self.output_pos

    def train(self):
        """ Main training loop
        """
        best_mrr, best_epoch = 0, 0
        self.sess.run(self.embeddings_init, feed_dict={self.emb_placeholder: self.index_to_embedding})
        tf.logging.info("Training {}".format(len(self.train_set[0])))
        for epoch in range(1, self.args.epochs + 1):
            num_batches = int(len(self.train_set[0]) / self.args.batch_size)
            all_acc = 0
            for i in range(0, num_batches + 1):
                batch = batchify(self.train_set[:-1], i, self.args.batch_size, max_sample=len(self.train_set[0]))
                if 0 == len(batch[0]):
                    continue
                feed_dict = self.get_feed_dict(batch)
                _, loss = self.sess.run([self.train_op, self.cost], feed_dict)
                all_acc += (loss * len(batch))

            if epoch % self.args.eval == 0:
                dev_mrr, dev_preds = self.evaluate(self.dev_set, self.args.batch_size, epoch, set_type='Dev')
                if dev_mrr > best_mrr:
                    self.saver.save(self.sess, self.ckpt_path)
                    best_mrr = dev_mrr
                    best_epoch = epoch
        tf.logging.info(f'Best epoch: {best_epoch} Best MRR: {best_mrr}')

    def evaluate(self, data, bsz, epoch, set_type=''):

        def print_results(str, result):
            tf.logging.info(str.ljust(20, ' ') + '  '.join(''.join(
                filter(lambda w: w != '<user>', [self.index_to_word[w] for w in result])
            ).split('9')))

        def to_str(li):
            return ''.join([str(i) for i in li])

        # import pdb; pdb.set_trace()
        questions, answers, labels = data[0], data[2], data[-1]

        num_batches = int(len(questions) / bsz)
        # num_batches = 5
        all_preds = []

        for i in tqdm(range(num_batches + 1)):
            batch = batchify(data, i, bsz, max_sample=len(questions))
            if len(batch) == 0:
                continue

            feed_dict = self.get_feed_dict(batch[:-1], mode='testing')
            loss, predictions = self.sess.run([self.cost, self.predict_op], feed_dict)
            all_preds.extend(predictions)

        di = defaultdict(list)
        for i, question in enumerate(questions):
            di[to_str(question)].append(i)

        a, p, rr = [], [], []
        for vals in di.values():
            all_preds = np.array(all_preds)
            preds_slice = all_preds[vals]
            preds_slice = [pred[0] for pred in preds_slice]
            max_idx = np.argmax(preds_slice)
            batch_offset = vals[0] + max_idx
            pred = answers[batch_offset]
            p.append(to_str(pred))
            act = labels[batch_offset]
            a.append(to_str(act))

            sorted_idx = np.argsort(preds_slice)[::-1]
            for i, idx in enumerate(sorted_idx):
                batch_offset = vals[0] + idx
                if answers[batch_offset] == labels[batch_offset]:
                    rr.append(1 / (i + 1))
                    break

        acc = accuracy_score(a, p)
        mrr = np.mean(rr)

        tf.logging.info('Epoch: {} {} P@1: {} MRR: {}'.format(epoch, set_type, acc, mrr))

        return mrr, all_preds

    def predict(self, data: Tuple) -> None:
        feed_dict = self.get_feed_dict(data, mode='predict')
        self.saver.restore(self.sess, self.ckpt_path)
        predictions = self.sess.run(self.predict_op, feed_dict=feed_dict)
        tf.logging.info(predictions)
        return predictions


def test_predict():

    num_neg = 3
    top_n = 10


    # question = 'what has been the status regarding creative commons since june 2012'
    # correct = 'it has been possible to select a creative commons license as the default , allowing other users to reuse and remix the material if it is free of copyright. .'

    question = 'how  do i convince parents to buy mr an ipod i have had a zen micro for about a year is that a decent amount of time'
    correct = 'do a bunch of extra chores first then after they tell you how much they appreciate all your hard work spring it on them and tell them how much it would mean to you to have a new ipod'


    tf.logging.info('start load answers')
    answers = pickle.load(open('/tmp/ans.pkl', 'rb'))
    answers = [a for a in answers if a != correct]
    tf.logging.info('end load answers')
    answers.append(correct)
    tf.logging.info('start prepare answers')
    ans = [[answers[0], 1]]
    # for a in answers[1:]:
    for a in answers[-num_neg:]:
        ans.append([a, 0])
    tf.logging.info('end prepare answers')
    tf.logging.info('start create_feed_data')
    data = dataset.create_feed_data({question: ans}, many=True)
    tf.logging.info('end create_feed_data')
    tf.logging.info('start predict')
    preds = hyper_qa.predict(data)
    preds = [p[0] for p in preds]
    ids = np.argsort(preds)[::-1]
    tf.logging.info('end predict')
    print(f"Running time: {timeit.timeit('lambda: hyper_qa.predict(data)', number=1)}")
    # tf.logging.info(idx)
    for idx in ids[:top_n]:
        tf.logging.info(data.pos_raw[idx])


if __name__ == '__main__':
    args = build_parser().parse_args()
    word_to_index, index_to_embedding = load_embedding_from_disks(args.glove, with_indexes=True)
    tf.logging.info('Embedding loaded')

    dataset = get_ds(args.dataset_name, args.dataset, word_to_index, index_to_embedding, args.qmax, args.amax,
                     args.char_min, args.num_neg)

    hyper_qa = HyperQA(dataset)
    tf.logging.info('HyperQA created')

    hyper_qa.train()

    # test_predict()
