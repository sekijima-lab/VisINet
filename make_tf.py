import numpy as np
import tensorflow as tf
import cv2
import sys
import re
import gc
import tarfile
import os
def make_example(image, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))

def main():
    CHANNELS = 3
    IMAGE_SIZE = 224
    NUM_CLASSES = 2
    IMAGE_MATRIX_SIZE = IMAGE_SIZE*IMAGE_SIZE*CHANNELS
    n=81
    argvs = sys.argv
    inputfile = argvs[1]
    outputfile = argvs[2]
    #tarfile.open(argvs[1] + "/" + argvs[1] + ".tar.gz",'r:gz', ignore_zeros=True).extractall(argvs[1])
    writer = tf.python_io.TFRecordWriter(outputfile)
    f = open(inputfile, 'r')
    line = f.readline()
    while line:
        line = line.rstrip()
        l = line.split()
        print(l[0])
        image = cv2.imread(l[0])
        print(image.shape)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image.flatten().astype(np.float32)
        label = np.array(int(l[1]))
        ex = make_example(image.tobytes(), label)
        writer.write(ex.SerializeToString())
        del image
        del label
        gc.collect()
        line = f.readline()

    f.close
    print("Done")

if __name__ == '__main__':
    main()
    # make_tf.py input(label) output(.tfrecord)
