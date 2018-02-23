import time
import collections
import random
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec)+ " sec"
    elif sec<(60*60):
        return str(sec/60)+ " min"

# Training source file with words
training_file = 'data.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1,])
    return content

training_data = read_data(training_file)
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary
dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)
print("Data set built...")


# Parameters
learning_rate = 0.001
#training_iters = 50000
training_iters = 500
#display_step = 1000
display_step = 100
n_input = 3
# Number of units in RNN cell
n_hidden = 512
# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])
# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(x, weights, biases):
    # reshape x for compatibility
    x = tf.reshape(x, [-1, n_input])
    # Convert input words to sequence of inputs
    # e.g. [Company] [size] [is] -> [650] [30] [45]
    x = tf.split(x, n_input, 1)
    #2-layer LSTM, each layer contains n_hidden units
    # Avarage Accuracy is 95% at 50K iterations. With 1-layer LSTM, accuracy is 90%...
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])
    #generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # there are n_inputs outputs, but we need only the last one
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)
# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
#Evaluation of the Model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print("Model evaluated...")


# Add ops to save and restore all the variables.
saver = tf.train.Saver()


# Variable initialization
init = tf.global_variables_initializer()

save_path = "/tmp/ties4911_t5_s{0}/model_{0}.ckpt".format(20000)

with tf.Session() as sess:
    saver.restore(sess, save_path)

    while True:
        prompt = "%s words: " % n_input
        sentance = input(prompt)
        sentance = sentance.strip()
        words= sentance.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = sess.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentance = "%s %s" % (sentance, reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentance)
        except:
            print("Word is not in dictionary")