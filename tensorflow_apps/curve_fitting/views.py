import json

import numpy as np
import tensorflow as tf
from django.http import HttpResponse, HttpResponseNotAllowed


def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs


def calculate(data):
    x_data = data.get('x_data')
    y_data = data.get('y_data')

    x_data_arr = np.linspace(-1, 1, len(x_data))[:, np.newaxis]
    y_data_arr = np.array(y_data).astype("float32")[:, np.newaxis]

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

    prediction = add_layer(l1, 10, 1, activation_function=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    prediction_value = []
    result = []

    for i in range(200):
        sess.run(train_step, feed_dict={xs: x_data_arr, ys: y_data_arr})
        prediction_value = sess.run(prediction, feed_dict={xs: x_data_arr})

    for item in prediction_value.tolist():
        for num in item:
            result.append(num)

    return result


def curve_fitting(request):
    if request.method == 'POST':
        data = json.loads(bytes.decode(request.body))
        response_data = {'x_data': data.get('x_data'), 'y_data':  calculate(data)}
        return HttpResponse(json.dumps(response_data), content_type="application/json")
    else:
        return HttpResponseNotAllowed(['POST'])
