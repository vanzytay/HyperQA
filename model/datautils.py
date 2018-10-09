import random
import pickle
import os


class YahooQA:

    def __init__(self, path):
        self.path = path
        self.dataset = dict()
        self.sample = dict()
        self.parts = ['test', 'train', 'dev']
        for part in self.parts:
            self.sample[part] = dict()
        self.feed_data = dict()

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

    def create_feed_data(self, dataset):
        for part in self.parts:

            questions, questions_len = [], []
            for question in dataset[part].keys():
               questions.append(question)
               questions_len.append(len(question))

            pos, pos_len, neg, neg_len = [], [], [], []
            for answer_tups in dataset[part].values():
                last = len(answer_tups) - 1
                first = random.randint(0, last - 1)
                assert answer_tups[last][1] == 1 and answer_tups[first][1] == 0
                pos_answer = answer_tups[last][0]
                pos.append(pos_answer)
                pos_len.append(len(pos_answer))
                neg_answer = answer_tups[first][0]
                neg.append(neg_answer)
                neg_len.append(len(neg_answer))
            self.feed_data[part] = questions, questions_len, pos, pos_len, neg, neg_len

        return self

    def display(self, part='train'):
        for tup in zip(*self.feed_data[part]):
            print(tup[0])
            print(tup[1])
            print(tup[2])
            print(tup[3])
            print(tup[4])
            print(tup[5])
            print('\n')
