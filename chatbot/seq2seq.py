#!/usr/bin/env python

import tensorflow as tf
import json
import numpy as np
import os, sys, re
import util.s2s_reader as s2s_reader

data_path = "data"
model_path = "output"

expression = r"[0-9]+|[']*[\w]+"

batch_size = 128

# parametri dei dati
bucket_option = [i for i in range(1, 200, 5)]
buckets = s2s_reader.create_bucket(bucket_option)

reader = s2s_reader.reader(file_name=data_path, batch_size=batch_size, buckets=buckets, bucket_option=bucket_option,
                           clean_mode=True)
vocab_size = len(reader.dict)

hidden_size = 512
projection_size = 300
embedding_size = 300
num_layers = 4

# dimensione dell'output per lo strato con funzione softmax
output_size = projection_size

truncated_std = 0.1
keep_prob = 0.95
max_epoch = 500
norm_clip = 5
adam_learning_rate = 0.05

model_name = "p" + str(projection_size) + "_h" + str(hidden_size) + "_x" + str(num_layers)
path = model_path + "/" + model_name

tf.reset_default_graph()
sess = tf.InteractiveSession()

enc_inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="enc_inputs")
targets = tf.placeholder(tf.int32, shape=(None, batch_size), name="targets")
dec_inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="dec_inputs")

emb_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=truncated_std), name="emb_weights")
enc_inputs_emb = tf.nn.embedding_lookup(emb_weights, enc_inputs, name="enc_inputs_emb")
dec_inputs_emb = tf.nn.embedding_lookup(emb_weights, dec_inputs, name="dec_inputs_emb")

encoder_cell_list = []
decoder_cell_list = []

for i in range(num_layers):
    cell = tf.nn.rnn_cell.LSTMCell(
        num_units=hidden_size,
        num_proj=projection_size,
        state_is_tuple=True
    )
    if i < num_layers - 1 or num_layers == 1:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
    encoder_cell_list.append(cell)
    decoder_cell_list.append(cell)

encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells=encoder_cell_list, state_is_tuple=True)
decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells=decoder_cell_list, state_is_tuple=True)

# definizione di encoder e decoder
_, encoder_states = tf.nn.dynamic_rnn(cell=encoder_cell,
                                      inputs=enc_inputs_emb,
                                      dtype=tf.float32,
                                      time_major=True,
                                      scope="encoder")

decoder_states, dec_states = tf.nn.dynamic_rnn(cell=decoder_cell,
                                               inputs=dec_inputs_emb,
                                               initial_state=encoder_states,
                                               dtype=tf.float32,
                                               time_major=True,
                                               scope="decoder")

# strati di output
output_q = tf.Variable(tf.truncated_normal(shape=[output_size, embedding_size], stddev=truncated_std), name="output_q")
output_b = tf.Variable(tf.constant(shape=[embedding_size], value=0.1), name="output_b")
softmax_q = tf.Variable(tf.truncated_normal(shape=[embedding_size, vocab_size], stddev=truncated_std), name="softmax_q")
softmax_b = tf.Variable(tf.constant(shape=[vocab_size], value=0.1), name="softmax_b")

decoder_states = tf.reshape(decoder_states, [-1, output_size], name="dec_ouputs")
dec_proj = tf.matmul(decoder_states, output_q) + output_b
logits = tf.nn.log_softmax(tf.matmul(dec_proj, softmax_q) + softmax_b, name="logits")

# funzione di loss
flat_targets = tf.reshape(targets, [-1])
total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=flat_targets)
avg_loss = tf.reduce_mean(total_loss)

# ottimizzazione tramite Adam
optimizer = tf.train.AdamOptimizer(adam_learning_rate)
gvs = optimizer.compute_gradients(avg_loss)
capped_gvs = [(tf.clip_by_norm(grad, norm_clip), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)

logit = logits[-1]
top_values, top_indexs = tf.nn.top_k(logit, k=10, sorted=True)

saver = tf.train.Saver()

os.makedirs(path)
sess.run(tf.global_variables_initializer())
losses = []


def update_summary(save_path, losses):
    summary_location = save_path + "/report.json"
    if os.path.exists(summary_location):
        os.remove(summary_location)
    with open(summary_location, 'w') as outfile:
        json.dump(losses, outfile)


def translate(token_list):
    enc = []
    for token in token_list:
        if token in reader.dict:
            enc.append(reader.dict[token])
        else:
            enc.append(reader.dict['[unk]'])
    # dec will be append with 2 inside the model
    print(enc)
    return enc


# Let's roll:

# Make a nice progress bar during training
from tqdm import tqdm

count = 0
epoch_loss = 0
epoch_count = 0

print("Addestramento in corso...")
pbar = tqdm(total=max_epoch)
pbar.set_description("Epoca 1: Perdita: 0 - Perdita media: 0 - Conteggio: 0")

while True:
    curr_epoch = reader.epoch
    data, index = reader.next_batch()
    enc_inp, dec_inp, dec_tar = s2s_reader.data_processing(data, buckets[index], batch_size)
    if reader.epoch != curr_epoch:
        losses.append(epoch_loss / epoch_count)
        epoch_loss = 0
        epoch_count = 0

        update_summary(path, losses)
        cwd = os.getcwd()
        saver.save(sess, path + "/model.ckpt")

        if reader.epoch == (max_epoch + 1):
            break

    feed_dict = {enc_inputs: enc_inp, dec_inputs: dec_inp, targets: dec_tar}
    _, loss_t = sess.run([train_op, avg_loss], feed_dict)
    epoch_loss += loss_t

    count += 1
    epoch_count += 1
    pbar.update(1)

    if count % 10 == 0:
        pbar.set_description("Epoch " + str(reader.epoch) + ": Perdita: " + str(loss_t) + " Perdita media: " + str(
            epoch_loss / epoch_count) + " Conteggio: " + str(epoch_count * batch_size))

print("Addestramento completo!")
