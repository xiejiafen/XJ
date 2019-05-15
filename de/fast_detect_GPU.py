import tensorflow as tf
from PIL import ImageDraw, Image
import cv2
from my_ops import conv, fc, max_pool, lrn, dropout
import numpy as np
import datetime
import os,sys


def fc2conv(x, num_in, num_out, filter_shape, name, relu=True):
    filter_height, filter_width, input_channels = filter_shape
    assert num_in == filter_height * filter_width * input_channels

    def convolve(i, k):
        return tf.nn.conv2d(i, k, strides=[1, 1, 1, 1], padding='VALID')

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05))
        biases = tf.get_variable('biases', [num_out], trainable=True, initializer=tf.constant_initializer(0.0))
        w_filter = tf.reshape(weights, [filter_height, filter_width, input_channels, num_out])
        act = convolve(x, w_filter) + biases

        if relu is True:
            relu_act = tf.nn.relu(act, name=scope.name)
            return relu_act
        else:
            return act


def full_conv_alex_net(detect_img):
    # Layer 1 (conv-relu-pool-lrn)
    print(detect_img.shape)
    conv1 = conv(detect_img, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
    pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
    norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

    # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
    conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
    pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
    norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    # 6th Layer: FC (w ReLu) -> Conv
    fc6_conv = fc2conv(pool5, 6 * 6 * 256, 4096, [6, 6, 256], name='fc6')

    # 7th Layer: FC (w ReLu) -> Conv
    fc7_conv = fc2conv(fc6_conv, 1 * 1 * 4096, 4096, [1, 1, 4096], name='fc7')

    # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
    fc8_conv = fc2conv(fc7_conv, 1 * 1 * 4096, 24, [1, 1, 4096], name='fc8', relu=False)

    return fc8_conv


def detect(fc8_conv, label=6, thres=0.9, draw_map=False):
    _, height, width, class_num = np.shape(fc8_conv)
    class_prob = tf.nn.softmax(fc8_conv).eval()
    points = np.argwhere(class_prob[0, :, :, label] > thres)

    # draw a probability heat map of certain label
    if draw_map is True:
        act_map = tf.reshape(class_prob[0, :, :, label], [height, width])
        act_map = tf.mul(act_map, 255)
        targets = tf.cast(act_map, tf.int32).eval()
        img = Image.fromarray(np.uint8(targets))
        img.save('activation_map_' + str(label) + '.jpg')

    return points


def detect_nms(fc8_conv, label=21 ,thres=0.5, draw_map=True):
    _, height, width, class_num = np.shape(fc8_conv)
    class_prob = tf.nn.softmax(fc8_conv).eval()
    class_prob = class_prob[0, :, :, label]
    #class_prob = class_prob[0, :, :, label]+class_prob[0, :, :, 13]+class_prob[0, :, :, 6]+class_prob[0, :, :, 4]
    points = np.argwhere(class_prob > thres)
    scores = class_prob[points.transpose().tolist()]

    # draw a probability heat map of certain label
    if draw_map is True:
        act_map = tf.reshape(class_prob, [height, width])
        act_map = tf.multiply(act_map, 255)
        targets = tf.cast(act_map, tf.int32).eval()
        img = np.uint8(targets)
        img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
        #cv2.imwrite("colormap3.jpg", img)

    return points, scores

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def main():
    os.chdir(os.path.split(os.path.abspath(sys.argv[0]))[0])
    f = open("./path.txt")
    lines = f.readlines()
    i=0
    while len(lines)!=0:
        test_tfrecords_dir=lines[i][:-1]
        #test_tfrecords_dir= sys.argv[1]
        img_src_bgr = cv_imread(test_tfrecords_dir)

        np_image_data = np.asarray(img_src_bgr)
        np_image_data = np_image_data[:, :, ::-1]  # bgr to rgb
        np_final = np.expand_dims(np_image_data, axis=0)
        img2 = tf.image.convert_image_dtype(np_final, tf.float32)

        with tf.variable_scope('model_definition') as scope:
            train_output = full_conv_alex_net(img2)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        file_write_pre = open(test_tfrecords_dir[:-4]+"_resul.txt", 'w')
        #file_write_pre = open("C:\\User\\DELL\\Desktop\\1\\_"+str(num)+"_resul.txt", 'w')

        with tf.Session() as sess:
            print('Init variable')
            sess.run(init)
            saver.restore(sess, "checkpoint/my-model.ckpt-200")

            print('detecting')
            starttime = datetime.datetime.now()

            with tf.variable_scope('model_definition') as scope:
                output = sess.run(train_output)
                points, scores = detect_nms(output, label=21, thres=0.5, draw_map=True)

                # non maximum suppression
                points_la = points * 32
                points_rb = points_la + 227
                boxes = np.hstack((points_la, points_rb))
                selected_indices = tf.image.non_max_suppression(boxes=boxes, scores=scores, max_output_size=points.shape[0],
                                                                iou_threshold=0.001)
                selected_boxes = tf.gather(boxes, selected_indices)

                # draw bounding boxes
                for box in selected_boxes.eval():
                    y1, x1, y2, x2 = box
                    cv2.rectangle(img_src_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    file_write_pre.writelines(str(box))
                    file_write_pre.write('\n')
                #cv2.imwrite(test_tfrecords_dir[:-4]+"_result.jpg", img_src_bgr)
                cv2.imencode(".jpg",img_src_bgr)[1].tofile(test_tfrecords_dir[:-4]+"_result.jpg")
            endtime = datetime.datetime.now()
            print((endtime - starttime).seconds)
        file_write_pre.close()
        tf.reset_default_graph()
        print("Finish!")
        i=i+1
    f.close()


if __name__ == '__main__':
    main()
