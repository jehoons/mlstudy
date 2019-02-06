from __future__ import division, print_function, absolute_import

import os, argparse
import numpy as np

import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt

from scipy.stats import norm
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
learning_rate = 0.001
num_steps = 30000
batch_size = 64

# Network Parameters
image_dim = 784 # MNIST images are 28x28 pixels
hidden_dim = 512
latent_dim = 2

progpath = os.path.dirname(os.path.realpath(__file__))
#model_data = os.path.join(progpath, 'model', 'model.ckpt') 
#os.system('mkdir -p %s' % os.path.dirname(model_data))

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Variables
weights = {
    'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim])),
    'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([hidden_dim, image_dim]))
}

biases = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'z_mean': tf.Variable(glorot_init([latent_dim])),
    'z_std': tf.Variable(glorot_init([latent_dim])),
    'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([image_dim]))
}

# Building the encoder
input_image = tf.placeholder(tf.float32, shape=[None, image_dim])
encoder = tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1']
encoder = tf.nn.tanh(encoder)

z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

tf.summary.histogram("z_mean", z_mean)
tf.summary.histogram("z_std", z_std)

# Sampler: Normal (gaussian) random distribution
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, 
    mean=0., stddev=1.0, name='epsilon')

# z_std = log(sigma^2) = 2log(sigma)
z = z_mean + tf.exp(z_std / 2) * eps # 1x2

tf.summary.histogram("z", z)

# Building the decoder (with scope to re-use these layers later)
decoder = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
decoder = tf.nn.tanh(decoder)
decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder)

# Define VAE Loss
def vae_loss(x_reconstructed, x_true):
    # reconstruction loss - cross entropy
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)

    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)
#    return tf.reduce_mean(kl_div_loss)

loss_op = vae_loss(decoder, input_image)
tf.summary.histogram("loss", loss_op)

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

def model_train(args_dict):
    mnist = input_data.read_data_sets("./scratch", one_hot=True)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() 

    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter( './logs/1/train ', sess.graph)    

        for i in range(1, num_steps+1):
            batch_x, _ = mnist.train.next_batch(batch_size)
            feed_dict = {input_image: batch_x}
            _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
                
            if i % 500 == 0 or i == 1:
                merge = tf.summary.merge_all()
                summary, _, l = sess.run([merge, train_op, loss_op], \
                        feed_dict=feed_dict)
                train_writer.add_summary(summary, i)

            if i % 100 == 0 or i == 1:
                print('Step %i, Loss: %f' % (i, l))
                
        saver.save(sess, args_dict['modeldata'])

def model_test(args_dict):
    init = tf.global_variables_initializer()

    with tf.Session() as sess: 
        sess.run(init)

        tf.train.Saver().restore(sess, args_dict['modeldata'])
        noise_input = tf.placeholder(tf.float32, shape=[None, latent_dim])

        decoder = tf.matmul(noise_input, weights['decoder_h1']) + biases['decoder_b1']
        decoder = tf.nn.tanh(decoder)
        decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
        decoder = tf.nn.sigmoid(decoder)

        n = 15
        x_axis = np.linspace(-3, 3, n)
        y_axis = np.linspace(-3, 3, n)

        canvas = np.empty((28 * n, 28 * n))
        for i, yi in enumerate(x_axis):
            for j, xi in enumerate(y_axis):
                z_mu = np.array([[xi, yi]] * batch_size)
                x_mean = sess.run(decoder, feed_dict={noise_input: z_mu})
                canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = \
                    x_mean[0].reshape(28, 28)

        plt.figure(figsize=(8, 10))
        Xi, Yi = np.meshgrid(x_axis, y_axis)
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()
        plt.savefig('scratch/latent.png', dpi=150)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("exec_mode", metavar="MODE", \
                        type=str, help="enter the mode for execution. i.e train|test")
    
    parser.add_argument("-l", "--logdir", metavar='LOG_DIR', \
                        type=str, help="log directory default: %(default)s", \
                        default="./logdir")

    parser.add_argument("-d", "--modeldata", metavar='MODEL_DIR', \
                        type=str, help="model directory default: %(default)s", \
                        default="scratch/model/VAE.ckpt")
    
    args = parser.parse_args()
    
    args_dict = vars(args)
    
    if args_dict['exec_mode'] == 'train': 
        model_train(args_dict)
    else: 
        model_test(args_dict)
        
