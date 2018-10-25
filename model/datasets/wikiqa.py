from datasets.Base import BaseQA
import pandas as pd
import numpy as np
np.random.seed(4242)
import os


class WikiQA(BaseQA):

    def __init__(self, path, word_to_index, index_to_embedding, qmax, amax, char_min, num_neg=5):
        super().__init__(path, word_to_index, index_to_embedding, qmax, amax, char_min, num_neg)

    def load_dataset(self, file_path):
        self.dataset = dict()
        train_path = os.path.join(file_path, 'WikiQA-train.tsv')
        dev_path = os.path.join(file_path, 'WikiQA-dev.tsv')
        test_path = os.path.join(file_path, 'WikiQA-test.tsv')
        cols = ['Question', 'Sentence', 'Label']
        self.dataset[self.Parts.train.name] = pd.read_csv(train_path, sep='\t')[cols]
        self.dataset[self.Parts.dev.name] = pd.read_csv(dev_path, sep='\t')[cols]
        self.dataset[self.Parts.test.name] = pd.read_csv(test_path, sep='\t')[cols]

    # def save_dataset(self, file_path):
    #     pickle.dump(self.dataset, open(file_path, 'wb'))

    def _create_splits(self):
        for part in self.parts:
            if part == self.Parts.train.name:
                self.splits[part] = self._create_feed_data(self.dataset[part], many=False)
            else:
                self.splits[part] = self._create_feed_data(self.dataset[part], many=True)

    def _create_feed_data(self, dataset, many=False):

        def to_ints(text, size, pad=0):
            # return text
            text_ints = [self.word_to_index[word] for word in text.split()]
            while len(text_ints) < size:
                text_ints.append(pad)
            return text_ints[:size]

        def get_pos_neg(answer_tups):
            pos_answers = []
            all_neg_answers = dataset.loc[dataset.Label == 0].Sentence.unique()
            neg_answers = np.random.choice(all_neg_answers, self.num_neg, replace=False)
            for idx, (answer, label) in answer_tups.iterrows():
                if label == 1:
                    pos_answers.append(answer)
                elif label == 0:
                    pass
                else:
                    raise ValueError('Neither pos nor neg value: {}'.format(label))
            result = []
            if many:
                if len(pos_answers) > 0 and len(neg_answers) > 0:
                    pos = pos_answers[0]
                    neg = neg_answers[0]
                    result.append([pos, neg, pos])
                    for neg in neg_answers:
                        result.append([neg, pos, pos])
            else:
                if len(pos_answers) > 0:
                    pos = pos_answers[0]
                    for neg in neg_answers:
                        result.append([pos, neg, pos])

            return result


        questions, questions_len, pos, pos_len, neg, neg_len, labels = [], [], [], [], [], [], []

        for question in dataset.Question.unique():
            answers = get_pos_neg(dataset[dataset.Question == question][['Sentence', 'Label']])
            for answer in answers:
                pos_answer, neg_answer, label = answer
                questions.append(to_ints(question, self.qmax))
                questions_len.append(len(question.split()))
                pos.append(to_ints(pos_answer, self.amax))
                pos_len.append(len(pos_answer.split()))
                neg.append(to_ints(neg_answer, self.amax))
                neg_len.append(len(neg_answer.split()))
                labels.append(to_ints(label, self.amax))

        return questions, questions_len, pos, pos_len, neg, neg_len, labels

    def display(self, part='train'):
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
    pass
