"""Held-out test evaluation matching the paper's compound-level protocol.

The paper aggregates a compound's 81 rendered views into a single
compound-level feature via 3D average pooling (averaging over the view AND
spatial dimensions), then ranks compounds by that single score using AUC and
EF1% (sklearn-computed), NOT per-image accuracy. In this codebase that 3D
pooling is implemented in resnet_v2_2.py (net = reduce_mean(net, [0,1,2],
...)), which is exactly what test4.py imports (`import resnet_v2_2 as
resnet_v2`) -- resnet_v2.py (used by mnist3.py's training/internal eval)
instead pools only over the spatial axes [1,2], i.e. per-image, so its
internal eval_results are an image-level training-set metric only, not
comparable to the paper's Table 1 numbers. Both files define the exact same
trainable variables (the patch only changes the reduce_mean axis, adding no
parameters), so a checkpoint trained via resnet_v2.py loads directly into
this resnet_v2_2.py-based graph.

Ground truth is read straight from each tfrecord's own `label` field
(embedded per-image by render_pack_worker.py), so no external
labels/compound-name file is needed here, unlike test4.py's CSV/jobname path
(which exists to attach compound identities for a separate downstream
ranking export, not needed just to get AUC/EF1%).

EF1%'s random-hitrate baseline uses ratio.py's nominal (active, all_) dataset
counts (ratio.counts(name)), not the empirical label ratio of the held-out
test split -- the actually-rendered/tested set is smaller than the nominal
dataset (docking/conformer-generation attrition, and only ~20% of shards are
held out for test), so its own label ratio is not the right denominator for
"how well do we do vs. picking randomly from the full screening population".

Usage: python3 eval_test.py <protein_name> <model_dir> <tfrecord> [<tfrecord> ...]
"""
from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

# script's own directory (sys.path[0], e.g. /work) takes priority so our
# ratio.py (with the counts() helper) shadows the older copy baked into the
# image; append /opt/visinet only as a fallback for resnet_v2_2/resnet_utils.
sys.path.append("/opt/visinet")
import ratio
import resnet_v2_2 as resnet_v2

slim = tf.contrib.slim
argvs = sys.argv
pos_weight = eval("ratio." + argvs[1])()

batch_size = 81  # one compound's 81 rendered views per batch


def parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.image.decode_png(features['image'], channels=3)
    label = features['label']
    image = tf.cast(image, tf.float32) * (1. / 255)
    image = tf.reshape(image, [224, 224, 3])
    label = tf.one_hot(label, 2)
    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, [2])
    return image, label


def enrichmentfactor(scores, label, random_hitrate, ratio=0.01):
    """EF1% per the paper: predicted_hitrate / random_hitrate, where
    random_hitrate is the active fraction of the FULL nominal dataset
    (ratio.counts(name)), not of whatever subset happens to be in the
    held-out test split -- the test split's active count is what we
    actually rank over (num_picked/predicted_hitrate), but the random
    baseline must reflect the real screening population composition."""
    num_picked = int(scores.size * ratio)
    order = np.argsort(scores)[::-1]
    top_label = label[order][:num_picked]
    predicted_hitrate = np.sum(top_label == 1) / num_picked
    return predicted_hitrate / random_hitrate


def main():
    model_dir = argvs[2]

    dataset = tf.data.TFRecordDataset(argvs[3:])
    dataset = dataset.map(parser)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    x_image = tf.reshape(features, [-1, 224, 224, 3])
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        nets, end_points = resnet_v2.resnet_v2_50(x_image, num_classes=2, is_training=False)
    prediction = end_points['predictions']  # 3D-avg-pooled: one row per compound
    true_label = tf.reduce_mean(labels, axis=0, keep_dims=True)

    saver = tf.train.Saver()
    scores = []
    truths = []
    with tf.Session() as sess:
        path = tf.train.latest_checkpoint(model_dir)
        print(f"restoring checkpoint: {path}")
        saver.restore(sess, path)
        sess.run(tf.local_variables_initializer())
        n = 0
        try:
            while True:
                pred_val, label_val = sess.run([prediction, true_label])
                scores.append(pred_val[0, 1])
                truths.append(int(np.argmax(label_val[0])))
                n += 1
                if n % 200 == 0:
                    print(f"{n} compounds scored", flush=True)
        except tf.errors.OutOfRangeError:
            print(f"Scoring done, {n} compounds total.")

    scores = np.array(scores)
    truths = np.array(truths)
    active_count, all_count = ratio.counts(argvs[1])
    random_hitrate = active_count / all_count
    auc = roc_auc_score(truths, scores)
    ef1 = enrichmentfactor(scores, truths, random_hitrate, ratio=0.01)
    acc = np.mean((scores >= 0.5).astype(int) == truths)
    print(f"n_compounds={len(scores)} n_pos={int(truths.sum())} n_neg={int((truths==0).sum())}")
    print(f"dataset(ratio.py): active={active_count} all={all_count} random_hitrate={random_hitrate:.6f}")
    print(f"HELD_OUT_TEST_RESULTS: AUC={auc:.4f} EF1%={ef1:.2f} accuracy(thresh=0.5)={acc:.4f}")


if __name__ == "__main__":
    main()
