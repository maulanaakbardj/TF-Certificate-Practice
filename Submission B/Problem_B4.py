# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
                 "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only",
                 "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
                 "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
                 "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                 "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
                 "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
                 "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"]
    content = []
    labels = []

    for label in bbc.category:
        labels.append(label)

    for con in bbc.text:
        for word in stopwords:
            token = " " + word + " "
            con = con.replace(token, " ")
            con = con.replace(" ", " ")
        content.append(con)

    train_content, test_content = content[:1780], content[445:]
    train_labels, test_labels = labels[:1780], labels[445:]

    train_content = np.array(train_content)
    test_content = np.array(test_content)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)  # YOUR CODE HERE
    tokenizer.fit_on_texts(train_content)

    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(train_content)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

    test_sequences = tokenizer.texts_to_sequences(test_content)
    test_padded = pad_sequences(test_sequences, maxlen=max_length, truncating=trunc_type)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    label_index = label_tokenizer.word_index

    label_sequences = np.array(label_tokenizer.texts_to_sequences(train_labels))

    test_label_sequences = np.array(label_tokenizer.texts_to_sequences(test_labels))

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 30
    history = model.fit(padded,
                        label_sequences,
                        epochs=epochs,
                        validation_data=(test_padded, test_label_sequences))

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.


if __name__ == '__main__':
    model = solution_B4()
    model.save("model_B4.h5")