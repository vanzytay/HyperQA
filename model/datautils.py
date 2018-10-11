import random
import pickle
import os
from enum import Enum


class YahooQA:
    class Parts(Enum):
        train = 1
        test = 2
        dev = 3

    def __init__(self, path):
        self.path = path
        self.dataset = dict()
        self.sample = dict()
        self.parts = list(self.Parts.__members__.keys())
        for part in self.parts:
            self.sample[part] = dict()
        # self.feed_data = dict()
        self.feed_data = tuple()

    def load_dataset(self, name):
        file_path = os.path.join(self.path, name)
        self.dataset = pickle.load(open(file_path, 'rb'))
        return self

    def save_dataset(self, dataset, name):
        file_path = os.path.join(self.path, name)
        pickle.dump(dataset, open(file_path, 'wb'))
        return self

    def create_sample(self, size=5):
        for part in self.parts:
            keys = list(self.dataset[part].keys())[:size]
            for key in keys:
                self.sample[part][key] = self.dataset[part][key]
        return self

    def create_feed_data(self, dataset, word_to_index, qmax, pos_max, neg_max):

        def to_ints(text, size, pad=0):
            text_ints = [word_to_index[word] for word in text]
            while len(text_ints) < size:
                text_ints.append(pad)
            return text_ints[:size]

        def get_pos_neg(answer_tups):
            pos_answers, neg_answers = [], []
            for tup in answer_tups:
                if tup[1] == 1:
                    pos_answers.append(tup)
                elif tup[1] == 0:
                    neg_answers.append(tup)
                else:
                    raise ValueError('Neither pos or neg value: {}'.format(tup[1]))

            # if len(pos_answers) > 1:
            #     print(pos_answers)
            #     raise ValueError('> 1 positive answers: '.format(len(pos_answers)))
            # if len(neg_answers) != len(answer_tups) - 1:
            #     raise ValueError('Wrong number of negative answers: '.format(len(neg_answers)))

            pos_answer = pos_answers[0]
            n = random.randint(0, len(neg_answers) - 1)
            neg_answer = neg_answers[n]

            try:
                assert pos_answer[1] == 1 and neg_answer[1] == 0
            except AssertionError as e:
                print(e)
                raise AssertionError(e)

            return pos_answer[0], neg_answer[0]

        questions, questions_len = [], []
        for question in dataset.keys():
            questions.append(to_ints(question, qmax))
            questions_len.append(len(question))

        pos, pos_len, neg, neg_len = [], [], [], []
        for answer_tups in dataset.values():
            pos_answer, neg_answer = get_pos_neg(answer_tups)
            pos.append(to_ints(pos_answer, pos_max))
            pos_len.append(len(pos_answer))
            # neg_answer = answer_tups[first][0]
            neg.append(to_ints(neg_answer, neg_max))
            neg_len.append(len(neg_answer))
        self.feed_data = questions, questions_len, pos, pos_len, neg, neg_len
        return self


    def display(self, part='train'):
        # for tup in zip(*self.feed_data[part]):
        for tup in zip(*self.feed_data):
            print(tup[0])
            print(tup[1])
            print(tup[2])
            print(tup[3])
            print(tup[4])
            print(tup[5])
            print('\n')
