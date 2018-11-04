import tensorflow as tf
import ops

def mobilenetv2(inputs, num_classes, is_train=True, reuse=False):
    exp = 6  # expansion ratio
    with tf.variable_scope('mobilenetv2'):
        net = ops.conv2d_block(inputs, 32, 3, 2, is_train, name='conv1_1')  # size/2

        net = ops.res_block(net, 1, 16, 1, is_train, name='res2_1')

        net = ops.res_block(net, exp, 24, 2, is_train, name='res3_1')  # size/4
        net = ops.res_block(net, exp, 24, 1, is_train, name='res3_2')

        net = ops.res_block(net, exp, 32, 2, is_train, name='res4_1')  # size/8
        net = ops.res_block(net, exp, 32, 1, is_train, name='res4_2')
        net = ops.res_block(net, exp, 32, 1, is_train, name='res4_3')

        net = ops.res_block(net, exp, 64, 1, is_train, name='res5_1')
        net = ops.res_block(net, exp, 64, 1, is_train, name='res5_2')
        net = ops.res_block(net, exp, 64, 1, is_train, name='res5_3')
        net = ops.res_block(net, exp, 64, 1, is_train, name='res5_4')

        net = ops.res_block(net, exp, 96, 2, is_train, name='res6_1')  # size/16
        net = ops.res_block(net, exp, 96, 1, is_train, name='res6_2')
        net = ops.res_block(net, exp, 96, 1, is_train, name='res6_3')

        net = ops.res_block(net, exp, 160, 2, is_train, name='res7_1')  # size/32
        net = ops.res_block(net, exp, 160, 1, is_train, name='res7_2')
        net = ops.res_block(net, exp, 160, 1, is_train, name='res7_3')

        net = ops.res_block(net, exp, 320, 1, is_train, name='res8_1', shortcut=False)

        net = ops.pwise_block(net, 1280, is_train, name='conv9_1')
        net = ops.global_avg(net)
        logits = ops.flatten(ops.conv_1x1(net, num_classes, name='logits'))

        pred = tf.nn.softmax(logits, name='prob')
        return logits, pred

def mobilenetv2_addBias(inputs, num_classes, channel_rito, is_train=True, reuse=False):
    exp = 6  # expansion ratio
    with tf.variable_scope('mobilenetv2'):
        net = ops.conv2d_block(inputs, round(32*channel_rito), 3, 2, is_train, name='conv1_1', bias=True)  # size/2

        net = ops.res_block(net, 1, round(16*channel_rito), 1, is_train, name='res2_1', bias=True)

        net = ops.res_block(net, exp, round(24*channel_rito), 1, is_train, name='res3_1', bias=True)  # size/2
        net = ops.res_block(net, exp, round(24*channel_rito), 1, is_train, name='res3_2', bias=True)

        net = ops.res_block(net, exp, round(32*channel_rito), 2, is_train, name='res4_1', bias=True)  # size/4
        net = ops.res_block(net, exp, round(32*channel_rito), 1, is_train, name='res4_2', bias=True)
        net = ops.res_block(net, exp, round(32*channel_rito), 1, is_train, name='res4_3', bias=True)

        net = ops.res_block(net, exp, round(64*channel_rito), 1, is_train, name='res5_1', bias=True)
        net = ops.res_block(net, exp, round(64*channel_rito), 1, is_train, name='res5_2', bias=True)
        net = ops.res_block(net, exp, round(64*channel_rito), 1, is_train, name='res5_3', bias=True)
        net = ops.res_block(net, exp, round(64*channel_rito), 1, is_train, name='res5_4', bias=True)

        net = ops.res_block(net, exp, round(96*channel_rito), 2, is_train, name='res6_1', bias=True)  # size/8
        net = ops.res_block(net, exp, round(96*channel_rito), 1, is_train, name='res6_2', bias=True)
        net = ops.res_block(net, exp, round(96*channel_rito), 1, is_train, name='res6_3', bias=True)

        net = ops.res_block(net, exp, round(160*channel_rito), 1, is_train, name='res7_1', bias=True)  # size/8
        net = ops.res_block(net, exp, round(160*channel_rito), 1, is_train, name='res7_2', bias=True)
        net = ops.res_block(net, exp, round(160*channel_rito), 1, is_train, name='res7_3', bias=True)

        net = ops.res_block(net, exp, round(320*channel_rito), 1, is_train, name='res8_1', bias=True,shortcut=False)

        net = ops.pwise_block(net, 128, is_train, name='conv9_1', bias=True)
        net = ops.global_avg(net)
        logits = ops.flatten(ops.conv_1x1(net, num_classes, name='logits', bias=True))

        pred = tf.nn.softmax(logits, name='prob')
        return logits, pred

