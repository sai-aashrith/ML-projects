import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import urllib.request
import json
sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]
# uploaded to github
tokenizer = Tokenizer(num_words = 50,oov_token='<oov>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= 50}
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences)
print(word_index)
print(sequences)
print(padded)
#
# Building and training a sentiment classifier on sarcasm
# import json
# import numpy as np
# import urllib


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    # Load the JSON dataset
    with open('sarcasm.json', 'r') as f:
        data = json.load(f)

    sentences = []
    labels = []
    for item in data:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    # Split the data into training and testing sets
    training_sentences = sentences[:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]

    # Tokenize the sentences and pad them to a fixed length
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Convert lists to numpy arrays
    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    # Create the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(training_padded, training_labels, epochs=30, validation_data=(testing_padded, testing_labels))

    return model


if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
