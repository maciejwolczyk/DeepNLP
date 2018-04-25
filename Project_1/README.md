# Sentence embeddings
## Author: Maciej Wołczyk

1. Usage

    Running LogReg.py without any arguments will run the training of the
skip-gram model with negative embeddings and PCA projection. (See: 2.
tested models).

2. Tested models
    1. Baseline model - very simple model which produces for any sentence an
       embedding of size 27 which contains the number of lowercase English
       letters in the sentence.
    2. Skip-gram model - as in word2vec, but without negative sampling and
       frequency sampling.
    3. Skip-gram model with negative sampling and PCA projection - as in
       word2vec but without frequency sampling and also projects the sentence
       on the first PCA component. Works best.
    4. C-BOW model with negative sampling - as in word2vec, but a special mode
       may also be set which uses the whole sentence as the context in training
       examples.

3. Hyperparameter search

    For each model a hyperparams search was conducted, using the adequatly
named function in the LogReg.py file. The best hyperparams are saved in the
Embedder.py file as global variables.

4. Possible improvements
    1. Frequency sampling may be added to the Skip-gram model
    2. Other ways of combining words into a sentence vector (not much
       specialized literature about it).
    3. Refactoring code to allow greater modularity (f.e. distinct
       preprocessing model-agnostic classes).
    4. Cross validation splits for better evaluation possibilites.
    5. Tune the frequency parameter from the Sanjeev et al. paper to maybe
       achieve better results?
