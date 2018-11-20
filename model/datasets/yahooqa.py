import random
from collections import namedtuple

from datasets.Base import BaseQA

random.seed(4242)
import pickle


class YahooQA(BaseQA):

    def __init__(self, path, word_to_index, index_to_embedding, qmax, amax, char_min, num_neg=5):
        super().__init__(path, word_to_index, index_to_embedding, qmax, amax, char_min, num_neg)

    def load_dataset(self, file_path):
        self.dataset = pickle.load(open(file_path, 'rb'))

    def save_dataset(self, file_path):
        pickle.dump(self.dataset, open(file_path, 'wb'))

    def _create_splits(self, shuffle=True):
        for part in self.parts:
            if part == self.Parts.train.name:
                if shuffle:
                    ds = self.dataset[part]
                    keys = list(ds.keys())
                    random.shuffle(keys)
                    shuffled = {key: ds[key] for key in keys}
                self.splits[part] = self.create_feed_data(shuffled, many=False)
            else:
                self.splits[part] = self.create_feed_data(self.dataset[part], many=True)

    def create_feed_data(self, dataset, many=False):

        def to_ints(text, size, pad=0):
            # return text
            text_ints = [self.word_to_index[word] for word in text.split()]
            while len(text_ints) < size:
                text_ints.append(pad)
            return text_ints[:size]

        def not_valid(text, min_chars=5, max_words=50):
            return len(text.split()) > max_words or len(text) < min_chars

        def get_all_neg_answers():
            all_neg_answers = []
            for answers in dataset.values():
                for answer in answers:
                    if not_valid(answer[0], min_chars=self.char_min, max_words=self.amax):
                        print('Invalid answer: {}'.format(answer[0]))
                        continue
                    if answer[1] == 0:
                        all_neg_answers.append(answer[0])
            all_neg_answers = list(set(all_neg_answers))
            all_neg_answers = list(zip(all_neg_answers, [0] * len(all_neg_answers)))
            return all_neg_answers

        def get_pos_neg(answer_tups, all_neg_answers):
            pos_answers = []
            # this is due to test_prediction getting otherwise sample_size > len(all_neg_answers)
            sample_size = min(self.num_neg, len(all_neg_answers))
            neg_answers = random.sample(all_neg_answers, sample_size)
            for tup in answer_tups:
                if not_valid(tup[0], min_chars=self.char_min, max_words=self.amax):
                    print('Invalid answer: {}'.format(tup[0]))
                    continue
                if tup[1] == 1:
                    pos_answers.append(tup)
                elif tup[1] == 0:
                    pass
                else:
                    raise ValueError('Neither pos nor neg value: {}'.format(tup[1]))
            result = []
            if many:
                pos = pos_answers[0]
                neg = neg_answers[0]
                result.append([pos[0], neg[0], pos[0]])
                for neg in neg_answers:
                    assert_labels(pos, neg)
                    result.append([neg[0], pos[0], pos[0]])
            else:
                pos = pos_answers[0]
                for neg in neg_answers:
                    assert_labels(pos, neg)
                    result.append([pos[0], neg[0], pos[0]])

            return result

        def assert_labels(pos_answer, neg_answer):
            try:
                assert pos_answer[1] == 1 and neg_answer[1] == 0
            except AssertionError as e:
                print(e)
                raise AssertionError(e)

        questions, questions_len, pos, pos_len, neg, neg_len, pos_raw, labels = [], [], [], [], [], [], [], []
        FeedData = namedtuple(
            'FeedData',
            ['questions', 'questions_len', 'pos', 'pos_len', 'neg', 'neg_len', 'pos_raw', 'labels']
        )

        all_neg_answers = get_all_neg_answers()

        for question, answer_tups in dataset.items():
            answers = get_pos_neg(answer_tups, all_neg_answers)
            for answer in answers:
                pos_answer, neg_answer, label = answer
                questions.append(to_ints(question, self.qmax))
                questions_len.append(len(question.split()))
                pos.append(to_ints(pos_answer, self.amax))
                pos_len.append(len(pos_answer.split()))
                pos_raw.append(pos_answer)
                neg.append(to_ints(neg_answer, self.amax))
                neg_len.append(len(neg_answer.split()))
                labels.append(to_ints(label, self.amax))

        return FeedData(questions=questions, questions_len=questions_len, pos=pos, pos_len=pos_len, neg=neg,
                        neg_len=neg_len, pos_raw=pos_raw, labels=labels)

    def display(self, part='train'):
        # for tup in zip(*self.feed_data[part]):
        for tup in zip(*self.feed_data):
            print(tup[0])
            print(tup[1])
            print(tup[2])
            print(tup[3])
            print(tup[4])
            print(tup[5])
            print(tup[6])
            print('\n')


if __name__ == '__main__':
    path = '/Users/svetlin/workspace/q-and-a/YahooQA_Splits/data/env.pkl'
    yahoo_qa = YahooQA(path, None, None, None, None, None)
    dev = yahoo_qa.splits[YahooQA.Parts.dev.name]
    pass
