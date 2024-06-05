import numpy
from tensorflow.python.framework import dtypes
import collections

Datasets = collections.namedtuple('datasets', ['train', 'test'])


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


class DataSet(object):

    def __init__(self,
                 images,
                 labels):

        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

        self._epoch_test = 0
        self._index_in_epoch_test = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def epoch_test(self):
        return self._epoch_test

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go the the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self._images[start:self._num_examples]
            label_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._images[start:end]
            label_new_part = self._labels[start:end]

            return numpy.concatenate((data_rest_part, data_new_part), axis = 0), numpy.concatenate((label_rest_part, label_new_part), axis = 0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

def read_data_sets(one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   ):
    """Construct a data set from a given directory path of the data"""

    test_images = numpy.load('/gs/hs0/tga-science/img_dock/npy/kif11_test_17_image_224.npy')
    test_labels = numpy.load('/gs/hs0/tga-science/img_dock/npy/kif11_test_17_label_224.npy')
    train_images = numpy.load('/gs/hs0/tga-science/img_dock/npy/kif11_train_17_image_224.npy')
    train_labels = numpy.load('/gs/hs0/tga-science/img_dock/npy/kif11_train_17_label_224.npy')
#akt1_test_17_image_28.npy
    if reshape:
        test_images = numpy.reshape(test_images, (-1, numpy.prod(test_images.shape[1:])))
        train_images = numpy.reshape(train_images, (-1, numpy.prod(train_images.shape[1:])))
    if one_hot:
        test_labels = dense_to_one_hot(test_labels, 2)
        train_labels = dense_to_one_hot(train_labels, 2)

    

    train = DataSet(train_images,
                    train_labels)

   

    test = DataSet(test_images,
                   test_labels)

    return Datasets(train=train, test=test)
