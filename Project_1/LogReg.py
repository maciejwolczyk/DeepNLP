import Embedder
import numpy as np
import itertools
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_data(filename):
    with open(filename) as opened_file:
        data = [tuple(line.split("\t")) for line in opened_file]
    return [datum[0] for datum in data], [int(datum[1]) for datum in data]

class LogisticRegressionModel(object):
    def __init__(self, embedder):
        # embedder to bedzie klasa ktora przez was bedzie napisana
        np.random.seed(123)
        self.embedder = embedder
        self.model = LogisticRegression()

    def fit(self, X, Y):
        # tutaj nastepuje uczenie embeddingu
        self.embedder.train_embeddings(X)
        embedded = [self.embedder.embed(x) for x in X]
        # upewnienie sie ze embedding ma staly wymiar.
        # Nie przejscie tej asercji oznacza niezaliczenie zadania
        assert(len(set(len(embedding) for embedding in embedded))==1)
        self.model.fit(embedded, Y)

    def predict(self, X):
        embedded = [self.embedder.embed(x) for x in X]
        # j.w.
        assert(len(set(len(embedding) for embedding in embedded))==1)
        return self.model.predict(embedded)

    def score(self, X_test, Y_test):
        assert(len(X_test)==len(Y_test))
        predictions = self.predict(X_test)
        matching = sum(y1==y2 for y1, y2 in zip(predictions, Y_test))
        # Maciej Wolczyk: A very minor change to avoid integer division in
        # Python2.
        return matching * 1.0/len(Y_test)


def hyperparameter_search():
    results_list = []
    epochs = [1]
    words_num = [500]
    embedding_dim = [2000]
    batch_size = [128]
    f = open("results.txt", "w")
    hyperparams_lists = [epochs, words_num, embedding_dim, batch_size]
    X, Y = load_data("train_data")
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y)
    for hyperparam_set in itertools.product(*hyperparams_lists):
        (Embedder.EPOCHS_NUM, Embedder.WORDS_NUM,
         Embedder.EMBEDDING_DIM, Embedder.BATCH_SIZE) = hyperparam_set
        
        embedder = Embedder.Embedder()
        lgm = LogisticRegressionModel(embedder)
        lgm.fit(x_train, y_train)

        train_score = lgm.score(x_train, y_train)
        valid_score = lgm.score(x_valid, y_valid)

        print("Results for {}".format(hyperparam_set))
        print("Score on training dataset\t{}".format(train_score))
        print("Score on validation dataset\t{}".format(valid_score))
        results_list += [{"Hyperparams": hyperparam_set,
                          "Training score": train_score,
                          "Valid score": valid_score}]
        f.write(str([hyperparam_set, train_score, valid_score]) + "\n")
        f.flush()

    best = max(results_list, key = lambda k: k["Valid score"])
    f.write(str(results_list))
    print(best)
    f.close()

hyperparameter_search()

