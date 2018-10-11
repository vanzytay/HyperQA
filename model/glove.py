import numpy as np
import tensorflow as tf
import chakin

import json
import os
from collections import defaultdict

# chakin.search(lang='English')
# Downloading Twitter.25d embeddings from Stanford:

CHAKIN_INDEX = 17
NUMBER_OF_DIMENSIONS = 25
SUBFOLDER_NAME = "glove.twitter.27B"

DATA_FOLDER = "embeddings"
ZIP_FILE = os.path.join(DATA_FOLDER, "{}.zip".format(SUBFOLDER_NAME))
ZIP_FILE_ALT = "glove" + ZIP_FILE[5:]  # sometimes it's lowercase only...
UNZIP_FOLDER = os.path.join(DATA_FOLDER, SUBFOLDER_NAME)
if SUBFOLDER_NAME[-1] == "d":
    GLOVE_FILENAME = os.path.join(UNZIP_FOLDER, "{}.txt".format(SUBFOLDER_NAME))
else:
    GLOVE_FILENAME = os.path.join(UNZIP_FOLDER, "{}.{}d.txt".format(SUBFOLDER_NAME, NUMBER_OF_DIMENSIONS))


def download_glove():
    global ZIP_FILE
    if not os.path.exists(ZIP_FILE) and not os.path.exists(UNZIP_FOLDER):
        # GloVe by Stanford is licensed Apache 2.0:
        #     https://github.com/stanfordnlp/GloVe/blob/master/LICENSE
        #     http://nlp.stanford.edu/data/glove.twitter.27B.zip
        #     Copyright 2014 The Board of Trustees of The Leland Stanford Junior University
        print("Downloading embeddings to '{}'".format(ZIP_FILE))
        chakin.download(number=CHAKIN_INDEX, save_dir='./{}'.format(DATA_FOLDER))
    else:
        print("Embeddings already downloaded.")
    if not os.path.exists(UNZIP_FOLDER):
        import zipfile
        if not os.path.exists(ZIP_FILE) and os.path.exists(ZIP_FILE_ALT):
            ZIP_FILE = ZIP_FILE_ALT
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            print("Extracting embeddings to '{}'".format(UNZIP_FOLDER))
            zip_ref.extractall(UNZIP_FOLDER)
    else:
        print("Embeddings already extracted.")


def load_embedding_from_disks(glove_filename, with_indexes=True):
    """
    Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionaries
    `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct
    `word_to_embedding_dict` dictionary mapping from a string to a numpy array.
    """
    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []
    else:
        word_to_embedding_dict = dict()

    with open(glove_filename, 'r', encoding='utf-8') as glove_file:
        for (i, line) in enumerate(glove_file):
            split = line.split(' ')
            word = split[0]
            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )
            if with_indexes:
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    _WORD_NOT_FOUND = [0.0] * len(representation)  # Empty representation for unknown words.
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        return word_to_embedding_dict


def word_info(word='hello'):
    word_idx = word_to_index[word]
    word_emb = index_to_embedding[word_idx]
    return word, word_idx, word_emb, np.linalg.norm(word_emb)


if __name__ == '__main__':
    print(GLOVE_FILENAME)
    word_to_index, index_to_embedding = load_embedding_from_disks(GLOVE_FILENAME, with_indexes=True)
    emb = tf.get_variable('Embedding', shape=index_to_embedding.shape, dtype=tf.float64,
                          initializer=tf.zeros_initializer, trainable=False)
    emb_ph = tf.placeholder(tf.float64, shape=index_to_embedding.shape)
    emb_init = emb.assign(emb_ph)
    ids_ph = tf.placeholder(tf.int64, shape=[None])
    emb_layer = tf.nn.embedding_lookup(params=emb, ids=ids_ph)

    with tf.Session() as sess:
        sess.run(emb_init, feed_dict={emb_ph: index_to_embedding})
        print(sess.run(emb_layer, feed_dict={ids_ph: [word_to_index[w] for w in ['hello']]}))

    print(word_info()[-2])
