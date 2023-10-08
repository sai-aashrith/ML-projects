import matplotlib.pyplot as plt
import pandas as pd
import keras
import numpy as np
import tensorflow as tf
import sys
from keras.models import load_model
# This is character text generation
# print(f"Python {sys.version}")
# gpu = (tf.config.list_physical_devices('GPU'))
# print("GPU is available", gpu)


file_path = 'shakespeare.txt'
text = open(file_path,'r').read()
vocab = sorted(set(text))
char_to_ind = {char:ind for ind,char in enumerate(vocab)}
ind_to_char = np.array(vocab)

encoded_text= np.array(list((char_to_ind[c] for c in text)))
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
seq_len=120
sequences=char_dataset.batch(seq_len+1,drop_remainder=True)
def create_seq_targets(seq):
    input_txt=seq[:-1]
    target_txt=seq[1:]
    return  input_txt,target_txt
dataset= sequences.map(create_seq_targets)
batch_size =128
buffer_size= 10000
dataset = dataset.shuffle(buffer_size).batch(batch_size,drop_remainder=True)
vocab_size = len(vocab)
embed_dim = 64
rnn_neurons = 1024
from keras.losses import sparse_categorical_crossentropy
def sparse_cat_loss(y_true,y_preds):
    return sparse_categorical_crossentropy(y_true,y_preds,from_logits=True)
from keras.models import Sequential
from keras.layers import Embedding,GRU,Dense
def create_model(vocab_size,embed_dim,rnn_neurons,batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size,embed_dim,batch_input_shape=[batch_size,None]))
    model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
    model.add(Dense(vocab_size))
    model.compile(optimizer='adam',loss=sparse_cat_loss)
    return model
model = create_model(vocab_size=vocab_size,embed_dim=embed_dim,rnn_neurons=rnn_neurons,batch_size=batch_size)
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
# print(example_batch_predictions , "# (batch_size, sequence_length, vocab_size)")
sampled_indices = tf.random.categorical(example_batch_predictions[0],num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
print(ind_to_char[sampled_indices])


model.fit(dataset,epochs=10)
model.save('shakespeare_txtgen.h5')
model = create_model(vocab_size,embed_dim,rnn_neurons,batch_size=1)
model.load_weights('shakespeare_txtgen.h5')
model.build(tf.TensorShape([1,None]))

def generate_text(model,start_seed,gen_size=500,temp=1.0):
    num_generate = gen_size
    input_eval = [char_to_ind[s] for s in start_seed]
    input_eval = tf.expand_dims(input_eval , 0)
    text_generated =[]
    temperature = temp
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions,0)
        predictions = predictions/temperature
        predicted_id = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id],0)
        text_generated.append(ind_to_char[predicted_id])
    return (start_seed + "".join(text_generated))

print(generate_text(model,"JULIET",gen_size=1000))
