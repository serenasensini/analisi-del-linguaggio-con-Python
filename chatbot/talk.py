#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, sys, re
import util.s2s_reader as s2s_reader

data_path = "data"
model_path = "output"

expression = r"[0-9]+|[']*[\w]+"

batch_size = 1

bucket_option = [i for i in range(1, 200, 5)]
buckets = s2s_reader.create_bucket(bucket_option)

reader = s2s_reader.reader(file_name=data_path, batch_size=batch_size, buckets=buckets, bucket_option=bucket_option,
                           clean_mode=True)
vocab_size = len(reader.dict)

hidden_size = 512
projection_size = 300
embedding_size = 300
num_layers = 1

output_size = projection_size

keep_prob = 0.95
beam_size = 10
top_k = 10
max_sequence_len = 20

model_name = "p" + str(projection_size) + "_h" + str(hidden_size) + "_x" + str(num_layers)
save_path = model_path + "/" + model_name

tf.reset_default_graph()
sess = tf.InteractiveSession()

encoder_inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="encoder_inputs")
targets = tf.placeholder(tf.int32, shape=(None, batch_size), name="targets")
decoder_inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="decoder_inputs")

emb_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size]), name="emb_weights")
encoder_inputs_emb = tf.nn.embedding_lookup(emb_weights, encoder_inputs, name="encoder_inputs_emb")
decoder_inputs_emb = tf.nn.embedding_lookup(emb_weights, decoder_inputs, name="decoder_inputs_emb")

enc_cell_list = []
dec_cell_list = []

for i in range(num_layers):
    single_cell = tf.nn.rnn_cell.LSTMCell(
        num_units=hidden_size,
        num_proj=projection_size,
        # initializer=tf.truncated_normal_initializer(stddev=truncated_std),
        state_is_tuple=True
    )
    if i < num_layers - 1 or num_layers == 1:
        single_cell = tf.nn.rnn_cell.DropoutWrapper(cell=single_cell, output_keep_prob=keep_prob)
    enc_cell_list.append(single_cell)
	dec_cell_list.append(single_cell)

enc_cell = tf.nn.rnn_cell.MultiRNNCell(cells=enc_cell_list, state_is_tuple=True)
dec_cell = tf.nn.rnn_cell.MultiRNNCell(cells=dec_cell_list, state_is_tuple=True)

_, encoder_states = tf.nn.dynamic_rnn(cell=enc_cell,
									  inputs=encoder_inputs_emb,
									  dtype=tf.float32,
									  time_major=True,
									  scope="encoder")

decoder_outputs, decoder_states = tf.nn.dynamic_rnn(cell=dec_cell,
													inputs=decoder_inputs_emb,
													initial_state=encoder_states,
													dtype=tf.float32,
													time_major=True,
													scope="decoder")

project_w = tf.Variable(tf.truncated_normal(shape=[output_size, embedding_size]), name="project_w")
project_b = tf.Variable(tf.constant(shape=[embedding_size], value=0.1), name="project_b")
softmax_w = tf.Variable(tf.truncated_normal(shape=[embedding_size, vocab_size]), name="softmax_w")
softmax_b = tf.Variable(tf.constant(shape=[vocab_size], value=0.1), name="softmax_b")

decoder_outputs = tf.reshape(decoder_outputs, [-1, output_size], name="dec_ouputs")
dec_proj = tf.matmul(decoder_outputs, project_w) + project_b
logits = tf.nn.log_softmax(tf.matmul(dec_proj, softmax_w) + softmax_b, name="logits")

logit = logits[-1]
top_values, top_indexs = tf.nn.top_k(logit, k=beam_size, sorted=True)

def build_input(sequence):
    dec_inp = np.zeros((1, len(sequence)))
    dec_inp[0][:] = sequence
    return dec_inp.T


def print_sentence(index_list):
    for index in index_list:
        sys.stdout.write(reader.id_dict[index])
        sys.stdout.write(' ')
    sys.stdout.write('\n')


def predict(enc_inp):
    sequence = [2]

    dec_inp = build_input(sequence)

    candidates = []
    options = []

    feed_dict = {encoder_inputs: enc_inp, decoder_inputs: dec_inp}
    values, indexs, state = sess.run([top_values, top_indexs, decoder_states], feed_dict)

    for i in range(len(values)):
        candidates.append([values[i], [indexs[i]]])

    best_sequence = None
    highest_score = -sys.maxint - 1

    while True:

        # print candidates
        for i in range(len(candidates)):

            sequence = candidates[i][1]
            score = candidates[i][0]

            if sequence[-1] == 3 or len(sequence) >= max_sequence_len:
                if score > highest_score:
                    highest_score = score
                    best_sequence = sequence
                continue

            dec_inp = build_input(sequence)

            feed_dict = {encoder_states: state, decoder_inputs: dec_inp}
            values, indexs = sess.run([top_values, top_indexs], feed_dict)

            for j in range(len(values)):
                new_sequence = list(sequence)
                new_sequence.append(indexs[j])
                options.append([score + values[j], new_sequence])

        options.sort(reverse=True)
        candidates = []

        for i in range(min(len(options), top_k)):
            if options[i][0] > highest_score:
                candidates.append(options[i])

        options = []
        if len(candidates) == 0:
            break

    return best_sequence[:-1]


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


saver = tf.train.Saver()
cwd = os.getcwd()
saver.restore(sess, cwd + "/" + save_path + "/model.ckpt")
print("\nModel restored.")

print("## Il Dottore e' pronto per ascoltarti: ##")

while True:
    try:
        line = sys.stdin.readline()
    except KeyboardInterrupt:
        print("\nSessione conclusa")
        break

    token_list = re.findall(expression, line.lower())

    sequence = translate(token_list)
    enc_inp = build_input(sequence[::-1])
    response = predict(enc_inp)
    sys.stdout.write('-->: ')
    print_sentence(response)
    print(' ')

sess.close()
