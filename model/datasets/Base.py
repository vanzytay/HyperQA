import random
from enum import Enum


class BaseQA:

    class Parts(Enum):
        train = 1
        test = 2
        dev = 3

    _splits = None

    def __init__(self, path, word_to_index, index_to_embedding, qmax, amax, char_min, num_neg=5):
        self.path = path
        self.word_to_index = word_to_index
        self.index_to_embedding = index_to_embedding
        self.qmax = qmax
        self.amax = amax
        self.char_min = char_min
        self.num_neg = num_neg
        self.dataset = dict()
        self.parts = list(self.Parts.__members__.keys())
        self.feed_data = dict()

    def load_dataset(self, file_path):
        raise NotImplemented

    def _create_splits(self, shuffle=True):
        raise NotImplemented

    def create_feed_data(self, dataset, many=False):
        raise NotImplemented

    @property
    def splits(self):
        if self._splits is None:
            self._splits = dict()
            self.load_dataset(self.path)
            for part in self.parts:
                if part == self.Parts.train.name:
                    ds = self.dataset[part]
                    keys = list(ds.keys())
                    random.shuffle(keys)
                    shuffled = {key: ds[key] for key in keys}
                    self._splits[part] = self.create_feed_data(shuffled, many=False)
                else:
                    self._splits[part] = self.create_feed_data(self.dataset[part], many=True)
        return self._splits
