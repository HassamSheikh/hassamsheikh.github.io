import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
tfd = tfp.distributions
#tf.enable_eager_execution()


w0 = 0.125
b0 = 5.
x_range = [-20, 60]


negloglik = lambda y, rv_y: -rv_y.log_prob(y)

def load_dataset(n=150, n_tst=150):
  np.random.seed(43)
  def s(x):
    g = (x - x_range[0]) / (x_range[1] - x_range[0])
    return 3 * (0.25 + g**2.)
  x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
  eps = np.random.randn(n) * s(x)
  y = (w0 * x * (1. + np.sin(x)) + b0) + eps
  x = x[..., np.newaxis]
  x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
  x_tst = x_tst[..., np.newaxis]
  return y, x, x_tst


y, x, x_tst = load_dataset()
# model = tf.keras.Sequential([
#   tf.keras.layers.Dense(1),
#   tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
# ])
# #
# # tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./logdir', write_graph=True)
# #
# # # Do inference.
# model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01), loss=negloglik)
# model.fit(x, y, epochs=1000, verbose=False, callbacks=[])
# # #
# # # # Profit.
# [print(np.squeeze(w.numpy())) for w in model.weights];
# yhat = model(x_tst)
# assert isinstance(yhat, tfd.Distribution)











x_train = tf.placeholder(tf.float32, shape=[None, 1])
y_train = tf.placeholder(tf.float32, shape=[None, 1])
out_nn = tf.layers.dense(x_train, units=1, kernel_initializer='glorot_uniform')
# import pdb; pdb.set_trace()
final_out = tfd.Normal(loc=out_nn, scale=1)
sample = final_out.mean()
loss = final_out.log_prob(y_train)
loss = tf.negative(loss)
loss = tf.reduce_mean(loss)
optimize = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())
    for _ in range(1000):
        sess.run(optimize, feed_dict={x_train:x, y_train:y.reshape(-1, 1)})
    ff = sess.run(sample, feed_dict={x_train:x_tst})
    print([var.eval() for var in tf.trainable_variables()])




print(ff)
plt.figure(figsize=[6, 1.5])  # inches
#plt.figure(figsize=[8, 5])  # inches
plt.plot(x, y, 'b.', label='observed');
plt.plot(x_tst, ff,'r', label='mean', linewidth=4);
plt.ylim(-0.,17);
plt.yticks(np.linspace(0, 15, 4)[1:]);
plt.xticks(np.linspace(*x_range, num=9));

ax=plt.gca();
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_smart_bounds(True)
#ax.spines['bottom'].set_smart_bounds(True)
plt.legend(loc='center left', fancybox=True, framealpha=0., bbox_to_anchor=(1.05, 0.5))
plt.show()
