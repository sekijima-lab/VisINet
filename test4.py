"""Usage: python test4.py <protein_name> <file_path> <jobname> <inputfile>
<protein_name> must match one of the functions defined in ratio.py.
"""
import socket
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import os
import data_set
import time
import ratio
import resnet_v2_2 as resnet_v2
import sys

slim = tf.contrib.slim
argvs = sys.argv

IMAGE_PIXELS=224
CHANNELS = 3
IMAGE_SIZE = 224
NUM_CLASSES = 2
IMAGE_MATRIX_SIZE = IMAGE_SIZE*IMAGE_SIZE*CHANNELS
batch_size = 81
num_epochs = 5
pos_weight = eval("ratio." + argvs[1])()
file_path = argvs[2]
os.makedirs(file_path, exist_ok=True)

def enrichmentfactor(scores, label, ratio=0.01):
   num_picked = int(scores.size * ratio)
   random_hitrate = np.sum(label == 1) / label.size
   order = np.argsort(scores)[::-1]
   top_label = label[order][:num_picked]
   predicted_hitrate = np.sum(top_label == 1) / num_picked
   return predicted_hitrate / random_hitrate


def main(_):

      x = tf.placeholder(tf.float32, [None, 150528])
      y_ = tf.placeholder(tf.float32, [None, 2])

      x_image = tf.reshape(x, [-1, 224, 224, 3])
      isTrain = tf.placeholder(tf.bool)
      with slim.arg_scope(resnet_v2.resnet_arg_scope()):
          nets, end_points = resnet_v2.resnet_v2_50(x_image, num_classes = 2, is_training = isTrain)
      path = tf.train.latest_checkpoint(file_path+"/model") 
      saver = tf.train.Saver()
      with tf.Session() as sess:
        saver.restore(sess, path)
        graph = tf.get_default_graph()

        test_queue = tf.train.string_input_producer([inputfile],
                                                    num_epochs = 1) # data is repeated and it raises OutOfRange when data is over
        test_reader = tf.TFRecordReader()
        _, test_serialized_exam = test_reader.read(test_queue)
     
        test_exam = tf.parse_single_example(
          test_serialized_exam,
          features={
              'image': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64)
              })
        test_image = tf.image.decode_png(test_exam['image'], channels=3)
        test_label = test_exam['label']
        #test_label = tf.decode_raw(test_exam['label'], tf.int64)
        test_image = tf.cast(test_image, tf.float32) * (1. / 255)
        test_image = tf.reshape(test_image, [150528])
        test_label = tf.one_hot(test_label, 2)
        test_label = tf.cast(test_label,tf.float64)
        test_label = tf.reshape(test_label,[2])
        test_batch_image, test_batch_label = tf.train.batch(
           [test_image, test_label],
           batch_size=batch_size)
        prediction = end_points['predictions']
        y = tf.reduce_mean(y_, axis = 0, keep_dims = True)
        accu = tf.argmax(y, 1)
        pred = tf.argmax(prediction, 1)

        array_correct = tf.equal(pred, accu)
        test_op = tf.reduce_sum(tf.cast(array_correct, tf.int32))
    
        sess.run(tf.initialize_local_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) # for data batching

        # test (evaluate) !!!
        num_true = 0
        b = np.empty((0,2),float)
        d = np.empty((0,1),float)
        try:
            num_test = 0
            while not coord.should_stop():
                array_image, array_label = sess.run(
                    [test_batch_image, test_batch_label])
                feed_dict = {
                    x: array_image,
                    y_: array_label,
                    isTrain:False
                }
                predict = sess.run(
                   prediction,
                feed_dict = feed_dict)

                b = np.concatenate((b,predict),axis = 0)
                d = np.append(d,predict[:,1])
                num_test +=  1
                if  num_test % 10 == 0:
                    print(f"{num_test} cpds finished.")


        except tf.errors.OutOfRangeError:
            print('Scoring done !')

        order = np.argsort(d)[::-1]
        d = np.sort(d)[::-1]
        data = open(f"{file_path}/labels/{jobname}", "r").readlines()
        test = []
        for line in data:
           cpd_name = "_".join(line.split("_")[:-1])
           cpd_name = cpd_name.split(".")[-1]
           test.append(cpd_name)
        test = np.array(list(set(test)))

        print(f"test: {test}")
        print(f"order: {order}")
        print(f"d: {d}")

        import pandas as pd
        result = pd.DataFrame({"cpd": test[order], "score": d})
        result.to_csv(f"{file_path}/result/result_{jobname}", index=None, header=None)
        print("Done")

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    #cnt = len(list(tf.python_io.tf_record_iterator("/gs/hs0/tga-science/img_dock/tf/" + argvs[1] + "_224_test.tfrecord")))
    #print("データ件数：{}".format(cnt))
    jobname = argvs[3]
    inputfile = argvs[4]
    tf.app.run()
