import json

import numpy as np
import tensorflow as tf
from django.http import HttpResponse


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def calculate(request):
    data = json.loads(bytes.decode(request.body))

    x_data = data.get('x_data')
    y_data = data.get('y_data')

    x_data_arr = np.linspace(-1, 1, len(x_data))[:, np.newaxis]
    y_data_arr = np.array(y_data).astype("float32")[:, np.newaxis]

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
    # add hidden layer
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    # add output layer
    prediction = add_layer(l1, 10, 1, activation_function=None)

    # the error between prediciton and real data
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # important step
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    prediction_value = []

    return_y = []

    for i in range(200):
        sess.run(train_step, feed_dict={xs: x_data_arr, ys: y_data_arr})
        prediction_value = sess.run(prediction, feed_dict={xs: x_data_arr})

    for list in prediction_value.tolist():
        for num in list:
            return_y.append(num)

    response_data = {'x_data': x_data, 'y_data': return_y}

    return HttpResponse(json.dumps(response_data), content_type="application/json")
