import tensorflow as tf


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    def convolve(i, k):
        return tf.nn.conv2d(i, k,
                            strides=[1, stride_y, stride_x, 1],
                            padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights',
                                  shape=[filter_height, filter_width, input_channels / groups, num_filters],
                                  trainable=True,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable('biases',
                                 shape=[num_filters],
                                 trainable=True,
                                 initializer=tf.constant_initializer(0.0))
        tf.summary.histogram(name+"/weights", weights)
        tf.summary.histogram(name+"/biases", biases)

        if groups == 1:
            conv_img = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        # split input and weights and convolve them separately
        else:
            input_groups = tf.split(value=x,num_or_size_splits=groups, axis=3)
            weight_groups = tf.split(value=weights, num_or_size_splits=groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv_img = tf.concat(values=output_groups,axis=3)

        bias = tf.reshape(tf.nn.bias_add(conv_img, biases), conv_img.get_shape().as_list())
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable('biases', [num_out], trainable=True, initializer=tf.constant_initializer(0.0))
        tf.summary.histogram(name+"/weights", weights)
        tf.summary.histogram(name+"/biases", biases)

        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu is True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
