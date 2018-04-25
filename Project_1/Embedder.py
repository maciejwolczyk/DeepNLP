import numpy as np
import tensorflow as tf
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from collections import defaultdict
from tqdm import tqdm
from sklearn.decomposition import PCA


# Hyperparams (may be set in LogReg.py)
EMBEDDING_DIM = 100
WORDS_NUM = 800
BATCH_SIZE = 18
EPOCHS_NUM = 25
WINDOW_SIZE = 4

stop_set = set(stopwords.words('english'))


def build_dictionary(data):
    word_dict = defaultdict(int)
    for sentence in data:
        for word in sentence:
            word_dict[word] += 1
    words_usage = sorted(word_dict.items(), key=lambda k: k[1], reverse=True)
    most_used = words_usage[:WORDS_NUM-1]
    top_word_dict = defaultdict(int)
    for k, v in most_used:
        top_word_dict[k] = v
    return top_word_dict


class BaselineEmbedder(object):
    """A very simple embedder for comparison purposes. It achieves about 53%
    accuracy in the test task."""
    def __init__(self):
        pass

    def train_embeddings(self, data):
        """No need to train so we just pass."""
        pass

    def embed(self, sentence):
        """
        Returns the embedding for the sentence
        as the sum of all letters
        """
        return [sentence.lower().count(c) for c in string.ascii_lowercase]


class SkipGramEmbedder(object):
    """Skip-gram embedder without negative sampling. Works pretty well, but
    training takes a very long time in comparison to the negative sampling
    version."""

    def __init__(self):
        pass

    def generate_training_examples(self, data):
        examples = []
        for sentence in data:
            for idx, word in enumerate(sentence):
                if word not in self.lookup_dict:
                    continue
                left_edge = max((idx - WINDOW_SIZE), 0)
                right_edge = min((idx + WINDOW_SIZE), len(sentence) - 1)
                context_indices = (list(range(left_edge, idx))
                                   + list(range(idx + 1, right_edge + 1)))
                examples += [(word, sentence[other])
                             for other in context_indices
                             if sentence[other] in self.lookup_dict]
        return examples

    def prepare_data(self, data):
        tt = TweetTokenizer()
        tokenised_sentences = [tt.tokenize(sentence)
                               for sentence in data]
        filtered_sentences = []
        for sentence in tokenised_sentences:
            filtered_sentences += [word for word in sentence
                                   if word not in stop_set]
        return filtered_sentences

    def train_embeddings(self, data):
        """
        The data is a list of sentences, each sentence is a single string.
        """
        preprocessed_data = self.prepare_data(data)
        self.word_dict = build_dictionary(preprocessed_data)

        self.lookup_dict = defaultdict(int)
        for idx, word in enumerate(self.word_dict.keys()):
            self.lookup_dict[word] = idx + 1

        examples = self.generate_training_examples(preprocessed_data)
        inputs = tf.placeholder(tf.int32, shape=[None])
        labels = tf.placeholder(tf.int32, shape=[None])
        hidden = tf.get_variable(
                "hidden_layer",
                shape=[WORDS_NUM, EMBEDDING_DIM],
                initializer=tf.truncated_normal_initializer())
        output_weights = tf.get_variable(
                "output_layer",
                shape=[EMBEDDING_DIM, WORDS_NUM],
                initializer=tf.truncated_normal_initializer())

        one_hot_inputs = tf.one_hot(inputs, WORDS_NUM)
        embs = tf.matmul(one_hot_inputs, hidden)
        logits = tf.matmul(embs, output_weights)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
        optimize = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(EPOCHS_NUM)):
                for _ in tqdm(range(len(examples) // BATCH_SIZE)):
                    indices = np.random.choice(len(examples), size=BATCH_SIZE)
                    batch = [examples[idx] for idx in indices]
                    batch_inputs = [self.lookup_dict[word] for word, _ in batch]
                    batch_labels = [self.lookup_dict[context]
                                    for _, context in batch]
                    sess.run((optimize,), feed_dict={inputs: batch_inputs,
                                                     labels: batch_labels})
            self.embeddings = sess.run(hidden)

    def embed(self, sentence):
        """
        Returns the embedding for the sentence
        as the sum over the embeddings of the words
        """
        processed_sentence = self.prepare_data([sentence])[0]
        embedded = []
        for word in processed_sentence:
            if word in self.lookup_dict:
                embedded += [self.embeddings[self.lookup_dict[word]]]
        return np.array(embedded).sum(axis=0)


class NegativeSkipGramEmbedder(object):
    """Skip-gram embedder with negative sampling, with word sums calculated
    according to the "Simple but tough-to-beat baseline" paper by Sanjeev et al.
    This class achieved the best results (~72%) on the tests on the validation
    dataset so it's used as the default embedder."""
    def __init__(self):
        pass

    def generate_training_examples(self, data):
        """Gather (word, context) pairs for the training."""
        examples = []
        for sentence in data:
            for idx, word in enumerate(sentence):
                if word not in self.lookup_dict:
                    continue
                left_edge = max((idx - WINDOW_SIZE), 0)
                right_edge = min((idx + WINDOW_SIZE), len(sentence) - 1)
                context_indices = (list(range(left_edge, idx))
                                   + list(range(idx + 1, right_edge + 1)))
                examples += [(word, sentence[other])
                             for other in context_indices
                             if sentence[other] in self.lookup_dict]
        return examples

    def prepare_data(self, data):
        """Since the Tweet Tokenizer and filtering weren't very helpful, we use
        some very simple methods."""
        tokenised_sentences = [sentence.split(" ") for sentence in data]
        return tokenised_sentences

    def prepare_sentence(self, sentence_list, alpha=1e-2):
        """Calculate the mean of the sentences. According to the Sanjeev et al.
        paper the mean should be weighted with inverse frequency, but on this
        dataset this approach wasn't succesful, so we omit this step."""
        sentence_vecs = []
        for sentence in sentence_list:
            word_vecs = []
            for word in sentence:
                # freq = self.word_dict[word] * 1.0 / self.words_sum
                freq = 0
                word_vecs += [self.embeddings[self.lookup_dict[word]]
                              * (alpha / (alpha + freq))]
            sentence_vecs += [np.array(word_vecs).mean(axis=0)]

        return sentence_vecs

    def sentence_to_vec(self, sentence_list):
        """Calculating the first component of sentences. New sentences will be
        projected on this component to get better results."""
        sentence_set = self.prepare_sentence(sentence_list)

        pca = PCA(n_components=EMBEDDING_DIM)
        pca.fit(np.array(sentence_set))
        u = pca.components_[0]
        u = u * u.T

        return u

    def train_embeddings(self, data):
        """
        The data is a list of sentences, each sentence is a single string.
        """

        # Data preprocessing.
        preprocessed_data = self.prepare_data(data)
        self.word_dict = build_dictionary(preprocessed_data)
        self.lookup_dict = defaultdict(int)
        for idx, word in enumerate(self.word_dict.keys()):
            self.lookup_dict[word] = idx + 1

        self.words_sum = sum(self.word_dict.values())
        examples = self.generate_training_examples(preprocessed_data)

        # Useful for repeated tests.
        tf.reset_default_graph()

        # Set up the model
        inputs = tf.placeholder(tf.int32, shape=[None])
        labels = tf.placeholder(tf.int32, shape=[None, 1])
        embeddings = tf.get_variable(
                "embeddings",
                [WORDS_NUM, EMBEDDING_DIM],
                initializer=tf.truncated_normal_initializer())
        embed = tf.nn.embedding_lookup(embeddings, inputs)

        nce_weights = tf.Variable(
            tf.truncated_normal(
                [WORDS_NUM, EMBEDDING_DIM],
                stddev=1.0 / np.sqrt(EMBEDDING_DIM)))
        nce_biases = tf.Variable(tf.zeros([WORDS_NUM]))
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=labels,
                inputs=embed,
                num_sampled=64,
                num_classes=WORDS_NUM))

        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=logits, labels=labels)
        optimize = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Train the model in minibatches.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(EPOCHS_NUM)):
                for _ in tqdm(range(len(examples) // BATCH_SIZE)):
                    indices = np.random.choice(len(examples), size=BATCH_SIZE)
                    batch = [examples[idx] for idx in indices]
                    batch_inputs = [self.lookup_dict[word] for word, _ in batch]
                    batch_labels = [[self.lookup_dict[context]]
                                    for _, context in batch]
                    sess.run(optimize, feed_dict={inputs: batch_inputs,
                                                  labels: batch_labels})
            self.embeddings = sess.run(embeddings)

        self.u = self.sentence_to_vec(preprocessed_data)

    def embed(self, sentence):
        """
        Returns the embedding for the sentence
        as the sum over the embeddings of the words
        """
        processed_sentence = self.prepare_data([sentence])[0]
        sentence_vec = self.prepare_sentence([processed_sentence])[0]

        return sentence_vec - self.u * sentence_vec


class NegativeCBOWEmbedder(object):
    """C-BOW embedder with negative sampling. Works in two modes.
    1) Classic C-BOW with window size given by a hyperparam.
    2) Sentence-specialized C-BOW inspired by paper "Unsupervised Learning of
    Sentence Embeddings using Compositional n-Gram Features". Due to the size
    of the training dataset only unigrams were used."""
    def __init__(self, sentence_mode=True):
        self.sentence_mode = sentence_mode

    def gen_sentence_samples(self, data):
        examples = []
        for sentence in data:
            for idx, word in enumerate(sentence):
                if word not in self.lookup_dict:
                    continue
                context_indices = (list(range(0, idx))
                                   + list(range(idx + 1, len(sentence))))
                context = [sentence[other] for other in context_indices]
                examples += [(context, word)]
        return examples

    def gen_window_samples(self, data):
        examples = []
        for sentence in data:
            for idx, word in enumerate(sentence[WINDOW_SIZE + 1:-WINDOW_SIZE]):
                if word not in self.lookup_dict:
                    continue
                left_edge = idx
                center = idx + WINDOW_SIZE + 1
                right_edge = idx + 2 * WINDOW_SIZE
                context_indices = (list(range(left_edge, center))
                                   + list(range(center + 1, right_edge + 1)))
                context = [sentence[other] for other in context_indices]
                examples += [(context, word)]
        return examples

    def prepare_data(self, data):
        tt = TweetTokenizer()
        tokenised_sentences = [tt.tokenize(sentence) for sentence in data]
        return tokenised_sentences

    def train_embeddings(self, unfiltered_data):
        """
        The data is a list of sentences, each sentence is a single string.
        """
        data = self.prepare_data(unfiltered_data)
        word_dict = build_dictionary(data)
        self.lookup_dict = defaultdict(int)
        for idx, word in enumerate(word_dict.keys()):
            self.lookup_dict[word] = idx

        examples = (self.gen_sentence_samples(data) if self.sentence_mode else
                    self.gen_window_samples(data))

        tf.reset_default_graph()

        inputs = tf.placeholder(tf.int32, shape=[None, None])
        labels = tf.placeholder(tf.int32, shape=[None, 1])
        embeddings = tf.get_variable(
                "embeddings",
                [WORDS_NUM, EMBEDDING_DIM],
                initializer=tf.truncated_normal_initializer())
        embed = tf.nn.embedding_lookup(embeddings, inputs)
        mean_embed = tf.reduce_mean(embed, axis=1)

        nce_weights = tf.Variable(
            tf.truncated_normal(
                [WORDS_NUM, EMBEDDING_DIM],
                stddev=1.0 / np.sqrt(EMBEDDING_DIM)))
        nce_biases = tf.Variable(tf.zeros([WORDS_NUM]))
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=labels,
                inputs=mean_embed,
                num_sampled=32,
                num_classes=WORDS_NUM))

        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #       logits=logits, labels=labels)
        optimize = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(EPOCHS_NUM)):
                for _ in tqdm(range(len(examples) // BATCH_SIZE)):
                    indices = np.random.choice(len(examples), size=BATCH_SIZE)
                    batch = [examples[idx] for idx in indices]
                    batch_inputs = [[self.lookup_dict[word] for word in context]
                                    for context, _ in batch]
                    max_len = max(len(b) for b in batch_inputs)
                    batch_inputs = [context + [0] * (max_len - len(context))
                                    for context in batch_inputs]
                    batch_labels = [[self.lookup_dict[word]]
                                    for _, word in batch]
                    sess.run(optimize, feed_dict={inputs: batch_inputs,
                                                  labels: batch_labels})
            self.embeddings = sess.run(embeddings)

    def embed(self, sentence):
        """
        Returns the embedding for the sentence
        as the sum over the embeddings of the words
        """
        processed_sentence = self.prepare_data([sentence])[0]
        sentence = [self.embeddings[self.lookup_dict[word]]
                    for word in processed_sentence]

        return np.array(sentence).sum(0)

# Negative skip-gram is the best model we've got so let's use it by default.
Embedder = NegativeSkipGramEmbedder
