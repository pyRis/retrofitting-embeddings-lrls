# coding: utf-8

import logging
import re
from collections import Counter

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset

# Hyperparameters
N_EMBEDDING = 300  # Dimensionality of the word vectors
BASE_STD = 0.01  # Standard deviation for initializing word vectors
BATCH_SIZE = (
    128  # Number of samples per batch -- try smaller batch sizes -- otherwise, too long
)
NUM_EPOCH = 100  # Number of epochs (iterations over the entire dataset)
MIN_WORD_OCCURENCES = (
    5  # Minimum number of occurrences for a word to be included in the vocabulary
)
X_MAX = 100  # Parameter controlling the scaling of the weighting function
ALPHA = 0.75  # Exponent in the weighting function
BETA = 0.0001  # Regularization parameter for the optimizer
RIGHT_WINDOW = 10  # Context window size for considering neighboring words on both sides
LEARNING_RATE = 0.001  # Learning rate for the optimizer

USE_CUDA = True

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


def cuda(x):
    if USE_CUDA:
        return x.cuda()
    return x


class WordIndexer:
    """Transform g a dataset of text to a list of index of words. Not memory
    optimized for big datasets"""

    def __init__(self, min_word_occurences=1, right_window=1, oov_word="OOV"):
        self.oov_word = oov_word
        self.right_window = right_window
        self.min_word_occurences = min_word_occurences
        self.word_to_index = {oov_word: 0}
        self.index_to_word = [oov_word]
        self.word_occurrences = {}
        self.re_words = re.compile(r"\b[a-zA-Z]{2,}\b")

    def _get_or_set_word_to_index(self, word):
        try:
            return self.word_to_index[word]
        except KeyError:
            idx = len(self.word_to_index)
            self.word_to_index[word] = idx
            self.index_to_word.append(word)
            return idx

    @property
    def n_words(self):
        return len(self.word_to_index)

    def fit_transform(self, texts):
        # use a tokenizer from XLM-R -- more suitable for different languages
        l_words = [list(self.re_words.findall(sentence.lower())) for sentence in texts]
        word_occurrences = Counter(word for words in l_words for word in words)

        self.word_occurrences = {
            word: n_occurences
            for word, n_occurences in word_occurrences.items()
            if n_occurences >= self.min_word_occurences
        }

        oov_index = 0
        return [
            [
                (
                    self._get_or_set_word_to_index(word)
                    if word in self.word_occurrences
                    else oov_index
                )
                for word in words
            ]
            for words in l_words
        ]

    def _get_ngrams(self, indexes):
        for i, target_index in enumerate(indexes):
            # Right context
            right_window = indexes[i + 1 : i + self.right_window + 1]
            for distance, context_index in enumerate(right_window):
                yield target_index, context_index, distance + 1
            # Left context
            left_window = indexes[max(0, i - self.right_window) : i]
            for distance, context_index in enumerate(reversed(left_window)):
                yield target_index, context_index, distance + 1

    def get_comatrix(self, data):
        comatrix = Counter()
        for indexes in data:
            l_ngrams = self._get_ngrams(indexes)
            for left_index, right_index, distance in l_ngrams:
                comatrix[(left_index, right_index)] += 1.0 / distance
        return zip(*[(left, right, x) for (left, right), x in comatrix.items()])


class GloveDataset(Dataset):
    def __len__(self):
        return self.n_obs

    def __getitem__(self, index):
        raise NotImplementedError()

    def __init__(self, texts, right_window=1, random_state=0):
        torch.manual_seed(random_state)

        self.indexer = WordIndexer(
            right_window=right_window, min_word_occurences=MIN_WORD_OCCURENCES
        )
        data = self.indexer.fit_transform(texts)
        left, right, n_occurrences = self.indexer.get_comatrix(data)
        n_occurrences = np.array(n_occurrences)
        self.n_obs = len(left)

        # We create the variables
        self.L_words = cuda(torch.LongTensor(left))
        self.R_words = cuda(torch.LongTensor(right))

        self.weights = np.minimum((n_occurrences / X_MAX) ** ALPHA, 1)
        self.weights = Variable(cuda(torch.FloatTensor(self.weights)))
        self.y = Variable(cuda(torch.FloatTensor(np.log(n_occurrences))))

        # We create the embeddings and biases
        N_WORDS = self.indexer.n_words
        L_vecs = cuda(torch.randn((N_WORDS, N_EMBEDDING)) * BASE_STD)
        R_vecs = cuda(torch.randn((N_WORDS, N_EMBEDDING)) * BASE_STD)
        L_biases = cuda(torch.randn((N_WORDS,)) * BASE_STD)
        R_biases = cuda(torch.randn((N_WORDS,)) * BASE_STD)
        self.all_params = [
            Variable(e, requires_grad=True)
            for e in (L_vecs, R_vecs, L_biases, R_biases)
        ]
        self.L_vecs, self.R_vecs, self.L_biases, self.R_biases = self.all_params


def gen_batchs(data):
    """Batch sampling function"""
    indices = torch.randperm(len(data))
    if USE_CUDA:
        indices = indices.cuda()
    for idx in range(0, len(data) - BATCH_SIZE + 1, BATCH_SIZE):
        sample = indices[idx : idx + BATCH_SIZE]
        l_words, r_words = data.L_words[sample], data.R_words[sample]
        l_vecs = data.L_vecs[l_words]
        r_vecs = data.R_vecs[r_words]
        l_bias = data.L_biases[l_words]
        r_bias = data.R_biases[r_words]
        weight = data.weights[sample]
        y = data.y[sample]
        yield weight, l_vecs, r_vecs, y, l_bias, r_bias


def get_loss(weight, l_vecs, r_vecs, log_covals, l_bias, r_bias):
    sim = (l_vecs * r_vecs).sum(1).view(-1)
    x = (sim + l_bias + r_bias - log_covals) ** 2
    loss = torch.mul(x, weight)
    return loss.mean()


def save_embeddings_txt(data: GloveDataset, filename="glove_embeddings.txt"):
    logging.info("Saving embeddings to %s", filename)
    L_vecs = data.L_vecs.cpu().detach().numpy()
    R_vecs = data.R_vecs.cpu().detach().numpy()
    with open(filename, "w") as f:
        for idx, word in enumerate(data.indexer.index_to_word):
            vec = (
                L_vecs[idx] + R_vecs[idx]
            )  # optionally average the two embeddings -- originally, sum
            vec_str = " ".join(map(str, vec))
            f.write(f"{word} {vec_str}\n")


def train_model(data: GloveDataset):
    optimizer = torch.optim.Adam(data.all_params, lr=LEARNING_RATE, weight_decay=1e-8)
    optimizer.zero_grad()

    min_loss = float("inf")
    best_epoch = 0

    for epoch in tqdm(range(NUM_EPOCH)):
        logging.info("Start epoch %i", epoch)
        num_batches = int(len(data) / BATCH_SIZE)
        avg_loss = 0.0

        for batch_idx, batch in enumerate(
            tqdm(gen_batchs(data), total=num_batches, mininterval=1)
        ):
            optimizer.zero_grad()
            loss = get_loss(*batch)
            avg_loss += loss.data / num_batches
            loss.backward()
            optimizer.step()

            # Check if current loss is the smallest encountered so far
            if loss.data < min_loss:
                min_loss = loss.data
                best_epoch = epoch

        logging.info("Average loss for epoch %i: %.5f", epoch + 1, avg_loss)

    logging.info("Best epoch: %i, Smallest loss: %.5f", best_epoch, min_loss)
    save_embeddings_txt(data, filename=f"glove_embeddings_best_epoch_{best_epoch}.txt")


if __name__ == "__main__":
    logging.info("Fetching data")
    dataset = load_dataset("cc100", "so")
    texts = dataset["train"]["text"]
    logging.info("Build dataset")
    glove_data = GloveDataset(texts, right_window=RIGHT_WINDOW)
    logging.info("#Words: %s", glove_data.indexer.n_words)
    logging.info("#Ngrams: %s", len(glove_data))
    logging.info("Start training")
    train_model(glove_data)
