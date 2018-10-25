from enum import Enum


class BaseQA:

    class Parts(Enum):
        train = 1
        test = 2
        dev = 3

    def __init__(self, path, word_to_index, index_to_embedding, qmax, amax, char_min, num_neg=5):
        self.path = path
        self.word_to_index = word_to_index
        self.index_to_embedding = index_to_embedding
        self.qmax = qmax
        self.amax = amax
        self.char_min = char_min
        self.num_neg = num_neg
        self.dataset = dict()
        self.splits = dict()
        self.parts = list(self.Parts.__members__.keys())
        for part in self.parts:
            self.splits[part] = dict()
        # self.feed_data = dict()
        self.feed_data = dict()
        self.load_dataset(self.path)
        self._create_splits()
