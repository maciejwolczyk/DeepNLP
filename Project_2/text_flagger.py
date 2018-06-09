import numpy as np
import tensorflow as tf
import tqdm
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
from math import ceil
from itertools import combinations

ONE_HOT_DIM = 125
TEXT_WIDTH = 125
BATCH_SIZE = 64
RNN_HIDDEN_DIM = 100
EPOCHS = 10
FC_NEURONS = 128


def calculate_f1_score(y_preds, y_trues):
    TP = TN = FP = FN = 0
    for y_pred, y_true in zip(y_preds, y_trues):
        if y_pred == 1 and y_true == 1:
            TP += 1
        elif y_pred == 0 and y_true == 0:
            TN += 1
        elif y_pred == 1 and y_true == 0:
            FP += 1
        else:
            FN += 1
    recall = 1.0 * TP / (1.0 * (TP + FN))
    precision = 1.0 * TP / (1.0 * (TP + FP))
    return 2.0 * (recall * precision) / (recall + precision)


class TextFlagger(object):
    def __init__(self):
        """
        initializer
        """
        (self.input_tensor, self.labels_tensor,
         self.logits_tensor, self.optimize_op) = self._build_model()

    def _build_model(self):
        optimizer = tf.train.AdamOptimizer()

        input_text = tf.placeholder(tf.int32, [None, TEXT_WIDTH])
        labels = tf.placeholder(tf.float32, [None, 1])
        one_hot = tf.one_hot(input_text, depth=ONE_HOT_DIM)

        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN_DIM)

        initial_state = rnn_cell.zero_state(
                tf.shape(input_text)[0],
                dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(
                rnn_cell, one_hot,
                initial_state=initial_state,
                dtype=tf.float32)

        first_conv = tf.layers.conv1d(
                outputs, 32, 3, 2,
                padding='same',
                activation=tf.nn.relu)
        second_conv = tf.layers.conv1d(
                first_conv, 64, 3,
                padding='same',
                activation=tf.nn.relu)
        third_conv = tf.layers.conv1d(
                second_conv, 16, 3, 2,
                padding='same',
                activation=tf.nn.relu)
        fourth_conv = tf.layers.conv1d(
                third_conv, 16, 2,
                padding='same',
                activation=tf.nn.relu)
        flat_fourth_conv = tf.layers.flatten(fourth_conv)
        fc_layer = tf.layers.dense(
                flat_fourth_conv, FC_NEURONS, activation=tf.nn.relu)
        fc_layer = tf.layers.dense(
                fc_layer, FC_NEURONS, activation=tf.nn.relu)
        fc_layer = tf.layers.dense(
                fc_layer, FC_NEURONS, activation=tf.nn.relu)
        logits = tf.layers.dense(fc_layer, 1)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits)
        optimize = optimizer.minimize(loss)

        return input_text, labels, logits, optimize

    def _prepare_dict(self, data):
        char_dict = defaultdict(int)
        for text in data:
            for letter in text:
                char_dict[letter] += 1

        most_used = sorted(
                char_dict.keys(),
                key=lambda x: char_dict[x],
                reverse=True)
        most_used = ['\0'] + list(most_used)
        most_used = most_used[:ONE_HOT_DIM]

        char_lookup = {char: idx for idx, char in enumerate(most_used)}

        return char_lookup

    def _process_sentences(self, X):
        processed_X = []
        for sentence in X:
            sentence = list(self.char_lookup[char]
                            for char in sentence
                            if char in self.char_lookup)

            # Handle length
            sentence = sentence[:TEXT_WIDTH]
            if len(sentence) < TEXT_WIDTH:
                diff = TEXT_WIDTH - len(sentence)
                sentence = sentence + [0] * diff

            processed_X += [sentence]
        return processed_X

    def fit(self, data, labels):
        """
        train on the given train data.
        data - list of messages as strings
        labels - list of numbers,
            0 means the message wasn't flagged
            1 menas it was
        """

        self.char_lookup = self._prepare_dict(data)
        X = self._process_sentences(data)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        batches_n = ceil(len(X) / BATCH_SIZE)

        for epoch in range(EPOCHS):
            for batch in tqdm.tqdm(range(batches_n)):
                X_batch = X[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
                y_batch = labels[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
                y_batch = [[y] for y in y_batch]
                logits,  _ = self.sess.run(
                    [self.logits_tensor, self.optimize_op],
                    feed_dict={self.input_tensor: X_batch,
                               self.labels_tensor: y_batch})


    def predict(self, data):
        """
        using the trained model, give its prediction on the given test data
        data - list of messages to flag or not as strings
        """

        X = self._process_sentences(data)

        batches_n = ceil(len(X) / 1024)

        logits_vals = []
        for batch in range(batches_n):
            X_batch = X[batch * 1024:(batch + 1) * 1024]

            logits_val = self.sess.run(
                    self.logits_tensor,
                    feed_dict={self.input_tensor: X_batch})
            logits_vals += logits_val.ravel().tolist()

        
        logits_vals = np.array(logits_vals)
        return (logits_vals > 0).astype("int32").ravel()


if __name__ == "__main__":
    X = []
    y = []
    with open("test.txt") as f:
        for idx, line in enumerate(f):
            sample, label = line.split("\t")
            X += [sample]
            y += [int(label)]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    text_flagger = TextFlagger()
    text_flagger.fit(X_train, y_train)
    y_preds = text_flagger.predict(X_test)
    score = calculate_f1_score(y_preds, y_test)
    print(score)
