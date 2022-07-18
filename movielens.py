# coding=utf-8
# Copyright 2021 The Self Aux Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiments on MovieLens Dataset."""

import collections
import pickle

from absl import app
import tensorflow as tf

TRAIN_DATASET_SIZE = 1600000
EVAL_DATASET_SIZE = 200000
DATA_PATH = ".../ml-1m"  # Specify path to MovieLens 1M dataset here.


def load_dataset(batch_size):
  with open(DATA_PATH + "/trainset_ds", "rb") as f:
    train_data = pickle.load(f, encoding="latin1")  # PY3 related.
  with open(DATA_PATH + "/evalset_ds", "rb") as f:
    valid_data = pickle.load(f, encoding="latin1")
  with open(DATA_PATH + "/testset_ds", "rb") as f:
    test_data = pickle.load(f, encoding="latin1")
  Dataset = collections.namedtuple("Dataset", ["train", "valid", "test"])
  return Dataset(
      tf.data.Dataset.from_tensor_slices(train_data).shuffle(
          TRAIN_DATASET_SIZE).batch(batch_size),
      tf.data.Dataset.from_tensor_slices(valid_data).shuffle(
          EVAL_DATASET_SIZE).batch(batch_size),
      tf.data.Dataset.from_tensor_slices(test_data).shuffle(
          EVAL_DATASET_SIZE).batch(batch_size))


def process_input(f):
  """Process input."""
  features = {
      "uid": f[0],
      "gender": f[1],
      "age": f[2],
      "occupation": f[3],
      "zip": f[4],
      "vid": f[5],
      "title": tf.string_split(f[6], " "),
      "genre": tf.string_split(f[7], "|")
  }
  labels = {
      "watch": f[8],
      "rating": f[9],
      "g_action": f[11],
      "g_adventure": f[12],
      "g_animation": f[13],
      "g_children": f[14],
      "g_comedy": f[15],
      "g_crime": f[16],
      "g_documentary": f[17],
      "g_drama": f[18],
      "g_fantasy": f[19],
      "g_filmnoir": f[20],
      "g_horror": f[21],
      "g_musical": f[22],
      "g_mystery": f[23],
      "g_romance": f[24],
      "g_scifi": f[25],
      "g_thriller": f[26],
      "g_war": f[27],
      "g_western": f[28]
  }
  return features, labels


def get_feature_columns():
  return [
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_hash_bucket("uid", 20000),
          40),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_hash_bucket("gender", 5),
          5),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_hash_bucket("age", 15), 10),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_hash_bucket(
              "occupation", 40), 20),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_hash_bucket("zip", 200),
          20),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_hash_bucket("vid", 10000),
          40),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_hash_bucket("title", 5000),
          20),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_hash_bucket("genre", 200),
          20)
  ]


def train(params):
  """The main train function."""
  if params.best_hp is not None:
    params = read_best_hp(params)
  tf.logging.info("params %s", params)
  shared_units = [int(i) for i in params.shared_units_str.split("_") if i]
  watch_units = [int(i) for i in params.watch_units_str.split("_") if i]
  rating_units = [int(i) for i in params.rating_units_str.split("_") if i]
  shared_layer = tf.keras.Sequential(
      [tf.keras.layers.DenseFeatures(get_feature_columns())] +
      [tf.keras.layers.Dense(u, tf.nn.relu) for u in shared_units])
  if params.dropout_rate > 0.0001:
    watch_tower = tf.keras.Sequential(
        [tf.keras.layers.Dense(u, tf.nn.relu) for u in watch_units] +
        [tf.keras.layers.Dropout(params.dropout_rate)] +
        [tf.keras.layers.Dense(1)])
    rating_tower = tf.keras.Sequential(
        [tf.keras.layers.Dense(u, tf.nn.relu) for u in rating_units] +
        [tf.keras.layers.Dropout(params.dropout_rate)] +
        [tf.keras.layers.Dense(1)])
  else:
    watch_tower = tf.keras.Sequential(
        [tf.keras.layers.Dense(u, tf.nn.relu) for u in watch_units] +
        [tf.keras.layers.Dense(1)])
    rating_tower = tf.keras.Sequential(
        [tf.keras.layers.Dense(u, tf.nn.relu) for u in rating_units] +
        [tf.keras.layers.Dense(1)])

  aux_watch_tower = tf.keras.Sequential([tf.keras.layers.Dense(1)])
  aux_rating_tower = tf.keras.Sequential([tf.keras.layers.Dense(1)])
  optimizer = tf.train.AdagradOptimizer(params.lr)
  if params.method == "uncertainty":
    watch_weight = tf.Variable(1.0)
    rating_weight = tf.Variable(1.0)

  def model(features, labels, get_layer=None):
    layer = shared_layer(features)
    if get_layer is not None:
      get_layer.append(layer)
    watch = watch_tower(layer)
    rating = rating_tower(layer)
    if params.aux_pool_len > 0:
      layer = tf.squeeze(
          tf.nn.pool(
              input=tf.expand_dims(layer, 2),
              window_shape=[params.aux_pool_len],
              pooling_type="AVG",
              strides=[params.aux_pool_len],
              padding="VALID"), 2)
    aux_watch = aux_watch_tower(layer)
    aux_rating = aux_rating_tower(layer)
    watch_loss = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.reshape(labels["watch"], (-1, 1)), logits=watch))
    rating_loss = tf.reduce_sum(
        tf.multiply(
            tf.reshape(labels["watch"], (-1, 1)),
            tf.math.square(tf.reshape(labels["rating"], (-1, 1)) - rating)))
    loss = params.alpha * watch_loss + (1 - params.alpha) * rating_loss
    aux_watch_loss = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.reshape(labels["watch"], (-1, 1)), logits=aux_watch))
    aux_rating_loss = tf.reduce_sum(
        tf.multiply(
            tf.reshape(labels["watch"], (-1, 1)),
            tf.math.square(tf.reshape(labels["rating"], (-1, 1)) - aux_rating)))
    aux_loss = (
        params.alpha * aux_watch_loss + (1 - params.alpha) * aux_rating_loss)
    if params.aux_loss_weight > 0:
      loss += params.aux_loss_weight * aux_loss
    watch_accu = tf.reduce_sum(
        tf.cast(
            tf.cast(tf.math.greater(watch, 0.0),
                    tf.float32) == tf.reshape(labels["watch"], (-1, 1)),
            tf.float32))
    return loss, watch_loss, rating_loss, watch_accu

  @tf.function()
  def plain_train_step(features, labels):
    loss, watch_loss, rating_loss, _ = model(features, labels)
    all_variables = (
        shared_layer.trainable_weights +  # Shared.
        watch_tower.trainable_weights +  # Watch.
        rating_tower.trainable_weights +  # Rating.
        aux_watch_tower.trainable_weights +  # AUX Watch.
        aux_rating_tower.trainable_weights)  # AUX Rating.
    gradients_and_var = optimizer.compute_gradients(loss, all_variables)
    optimizer.apply_gradients(gradients_and_var)
    return watch_loss, rating_loss

  @tf.function()
  def uncertainty_train_step(features, labels):
    """Uncertainty-reweighting method: https://arxiv.org/pdf/1705.07115.pdf."""
    _, watch_loss, rating_loss, _ = model(features, labels)
    watch_w = tf.clip_by_value(watch_weight, 0.01, 10.0)
    watch_loss = watch_loss / tf.exp(2 * watch_w)
    rating_w = tf.clip_by_value(rating_weight, 0.01, 10.0)
    rating_loss = rating_loss / (2 * tf.exp(2 * rating_w))
    new_loss = watch_loss + rating_loss + watch_w + rating_w
    all_variables = (
        shared_layer.trainable_weights +  # Shared.
        watch_tower.trainable_weights +  # Watch.
        rating_tower.trainable_weights +  # Rating.
        aux_watch_tower.trainable_weights +  # AUX Watch.
        aux_rating_tower.trainable_weights +  # AUX Rating.
        [watch_weight, rating_weight])  # Uncertainty weights.
    gradients_and_var = optimizer.compute_gradients(new_loss, all_variables)
    optimizer.apply_gradients(gradients_and_var)
    return watch_loss, rating_loss

  @tf.function()
  def mgda_train_step(features, labels):
    """MGDA-UB algorithm proposed in: https://arxiv.org/abs/1810.04650."""
    get_layer = []
    _, watch_loss, rating_loss, _ = model(features, labels, get_layer)

    # MGDA-UB, loss rescaling to balance classification and regression loss.
    watch_grad = optimizer.compute_gradients(params.alpha * watch_loss,
                                             get_layer[0])[0][0]
    rating_grad = optimizer.compute_gradients((1 - params.alpha) * rating_loss,
                                              get_layer[0])[0][0]
    if params.mgda_spec == "reduce_mean":
      watch_grad = tf.reduce_mean(watch_grad, 0)
      rating_grad = tf.reduce_mean(rating_grad, 0)
    elif params.mgda_spec == "reduce_sum":
      watch_grad = tf.reduce_sum(watch_grad, 0)
      rating_grad = tf.reduce_sum(rating_grad, 0)
    else:
      watch_grad = tf.reshape(watch_grad, [-1])
      rating_grad = tf.reshape(rating_grad, [-1])
    weight = tf.stop_gradient(
        tf.matmul(
            tf.reshape(watch_grad - rating_grad,
                       (1, -1)), tf.reshape(watch_grad, (-1, 1))) /
        tf.reduce_sum(tf.square(watch_grad - rating_grad)))
    weight = tf.clip_by_value(weight, 0.001, 0.999)  # value used in paper.

    new_loss = weight * watch_loss + (1 - weight) * rating_loss
    all_variables = (
        shared_layer.trainable_weights +  # Shared.
        watch_tower.trainable_weights +  # Watch.
        rating_tower.trainable_weights)  # Rating.
    gradients_and_var = optimizer.compute_gradients(new_loss, all_variables)
    optimizer.apply_gradients(gradients_and_var)
    return watch_loss, rating_loss

  @tf.function()
  def eval_step(features, labels):
    _, watch_loss, rating_loss, watch_accu = model(features, labels)
    Eval = collections.namedtuple("Eval",
                                  ["watch_loss", "rating_loss", "watch_accu"])
    return Eval(watch_loss, rating_loss, watch_accu)

  dataset = load_dataset(params.batch_size)

  for current_epoch in range(params.num_epochs):
    tf.logging.info("epoch: {}".format(current_epoch))
    epoch_watch_loss, epoch_rating_loss = 0, 0
    for f in dataset.train:
      features, labels = process_input(f)
      if params.method == "mtl":
        watch_loss, rating_loss = plain_train_step(features, labels)
      if params.method == "uncertainty":
        watch_loss, rating_loss = uncertainty_train_step(features, labels)
      if params.method == "mgda":
        watch_loss, rating_loss = mgda_train_step(features, labels)
      epoch_watch_loss += watch_loss / TRAIN_DATASET_SIZE
      epoch_rating_loss += rating_loss / TRAIN_DATASET_SIZE
    tf.logging.info("total train loss: %s || watch loss: %s || rating loss: %s",
                    epoch_watch_loss + epoch_rating_loss, epoch_watch_loss,
                    epoch_rating_loss)

    epoch_eval = {"watch_loss": 0, "rating_loss": 0, "watch_accu": 0}
    for f in dataset.valid if params.use_valid else dataset.test:
      features, labels = process_input(f)
      watch_loss, rating_loss, watch_accu = eval_step(features, labels)
      epoch_eval["watch_loss"] += watch_loss / EVAL_DATASET_SIZE
      epoch_eval["rating_loss"] += rating_loss / EVAL_DATASET_SIZE
      epoch_eval["watch_accu"] += watch_accu / EVAL_DATASET_SIZE
    tf.logging.info(
        "eval loss: %s || watch_loss: %s || rating_loss: %s || watch_accu: %s",
        epoch_eval["watch_loss"] + epoch_eval["rating_loss"],
        epoch_eval["watch_loss"], epoch_eval["rating_loss"],
        epoch_eval["watch_accu"])
  tf.logging.info("final eval loss %s",
                  epoch_eval["watch_loss"] + epoch_eval["rating_loss"])
  return epoch_eval


def read_best_hp(params):
  """Read best hyperparameters."""
  if params.best_hp == "aux":
    params.lr = 0.1
    params.aux_loss_weight = 1.0
    params.aux_pool_len = 0
    params.method = "mtl"
  elif params.best_hp == "aux_pool":
    params.lr = 0.1
    params.aux_loss_weight = 0.5
    params.aux_pool_len = 20
    params.method = "mtl"
  elif params.best_hp == "mtl":
    params.lr = 0.1
    params.aux_loss_weight = 0
    params.aux_pool_len = 0
    params.method = "mtl"
  elif params.best_hp == "uncertainty":
    params.lr = 0.1
    params.aux_loss_weight = 0
    params.aux_pool_len = 0
    params.method = "uncertainty"
  elif params.best_hp == "mgda":
    params.lr = 0.05
    params.aux_loss_weight = 0
    params.aux_pool_len = 0
    params.method = "mgda"
    params.mgda_spec = "reduce_sum"
  return params


def hparams():
  # For local run.
  return tf.HParams(
      batch_size=100,
      num_epochs=2,
      use_valid=True,
      lr=0.1,
      alpha=0.5,
      shared_units_str="200",
      watch_units_str="200",
      rating_units_str="200",
      aux_loss_weight=0.0,
      aux_pool_len=0,
      method="mtl",
      mgda_spec="default",
      best_hp="aux",
      dropout_rate=0.5)


def main(argv):
  del argv  # Unused.

  master_hparams = hparams()

  tf.reset_default_graph()
  final_eval = train(master_hparams)
  tf.logging.info("final results: %s", final_eval)
  return


if __name__ == "__main__":
  app.run(main)
