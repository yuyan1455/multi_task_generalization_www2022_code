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

"""Experiments on MultiMNIST and MultiFashion Dataset."""

import collections
import pickle

from absl import app
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

TRAIN_DATASET_SIZE = 100000
EVAL_DATASET_SIZE = 20000
METRICS_AVERAGE = 5
DATASET = 'MultiMNIST'  # One of ['MultiMNIST', 'MultiFashion'].
BATCH_SIZE = 256
USE_VALID = True  # If True, use validation dataset; else use test dataset.
# Specify the path to the MultiMNIST dataset here.
DATA_PATH_MULTIMNIST = '.../multi_mnist.pickle'
# Specify the path to the MultiFashion dataset here.
DATA_PATH_MULTIFASHION = '.../multi_fashion.pickle'

# Globs for uncertainty weighing.
l_weight = r_weight = l_l0_loss = r_l0_loss = l_uncertainty = r_uncertainty = None


def read_best_hp(params):
  """Return the best hyperparameters used for each method."""
  if DATASET == 'MultiMNIST':
    if params.best_hp == 'mtl':
      params.lr = 0.0008
      params.method = 'mtl'
    elif params.best_hp == 'aux':
      params.lr = 0.0006
      params.gamma = 20
      params.T_aux = 200
      params.method = 'aux'
    elif params.best_hp == 'aux_neck':
      params.lr = 0.0006
      params.gamma = 20
      params.neck = 8
      params.T_aux = 200
      params.method = 'aux_neck'
    elif params.best_hp == 'mgda':
      params.lr = 0.003
      params.method = 'mgda'
      params.mgda_spec = 'reduce_mean'
    elif params.best_hp == 'uncertainty':
      params.lr = 0.001
      params.method = 'uncertainty'
    elif params.best_hp == 'pta2':
      params.lr = 0.0006
      params.method = 'pta2'
    elif params.best_hp == 'pta3':
      params.lr = 0.0011
      params.method = 'pta3'
    elif params.best_hp == 'pta4':
      params.lr = 0.0004
      params.method = 'pta4'
    elif params.best_hp == 'aux_same':
      params.lr = 0.0011
      params.method = 'aux_same'
    return params
  elif DATASET == 'MultiFashion':
    if params.best_hp == 'mtl':
      params.lr = 0.0008
      params.method = 'mtl'
    elif params.best_hp == 'aux':
      params.lr = 0.0007
      params.gamma = 30
      params.T_aux = 300
      params.method = 'aux'
    elif params.best_hp == 'aux_neck':
      params.lr = 0.0007
      params.gamma = 30
      params.neck = 9
      params.T_aux = 300
      params.method = 'aux_neck'
    elif params.best_hp == 'mgda':
      params.lr = 0.0007
      params.method = 'mgda'
      params.mgda_spec = 'default'
    elif params.best_hp == 'uncertainty':
      params.lr = 0.001
      params.method = 'uncertainty'
    elif params.best_hp == 'pta2':
      params.lr = 0.0007
      params.method = 'pta2'
    elif params.best_hp == 'pta3':
      params.lr = 0.0011
      params.method = 'pta3'
    elif params.best_hp == 'pta4':
      params.lr = 0.0005
      params.method = 'pta4'
    elif params.best_hp == 'aux_same':
      params.lr = 0.0007
      params.method = 'aux_same'
    return params


def load_dataset(batch_size, dataset='MultiMNIST'):
  """Load datasets."""
  if dataset == 'MultiMNIST':  # Specify the path to the dataset here.
    path = DATA_PATH_MULTIMNIST
  elif dataset == 'MultiFashion':
    path = DATA_PATH_MULTIFASHION

  with open(path, 'rb') as f:
    trainx, trainy, testx, testy = pickle.load(f)
  trainx, testx = np.expand_dims(
      trainx, axis=-1).astype(np.float32), np.expand_dims(
          testx, axis=-1).astype(np.float32)

  valid_indices = [i * 6 for i in range(20000)]
  validx = trainx[valid_indices]
  validy = trainy[valid_indices]
  trainx = np.delete(trainx, valid_indices, axis=0)
  trainy = np.delete(trainy, valid_indices, axis=0)

  trainyl = tf.keras.utils.to_categorical(trainy[:, 0], num_classes=10)
  trainyr = tf.keras.utils.to_categorical(trainy[:, 1], num_classes=10)
  testyl = tf.keras.utils.to_categorical(testy[:, 0], num_classes=10)
  testyr = tf.keras.utils.to_categorical(testy[:, 1], num_classes=10)
  validyl = tf.keras.utils.to_categorical(validy[:, 0], num_classes=10)
  validyr = tf.keras.utils.to_categorical(validy[:, 1], num_classes=10)

  train_dataset = tf.data.Dataset.from_tensor_slices((trainx, trainyl, trainyr))
  train_dataset = train_dataset.shuffle(
      buffer_size=trainx.shape[0], seed=0,
      reshuffle_each_iteration=True).batch(batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices(
      (testx, testyl, testyr)).batch(batch_size)
  valid_dataset = tf.data.Dataset.from_tensor_slices(
      (validx, validyl, validyr)).batch(batch_size)

  Dataset = collections.namedtuple('Dataset', ['train', 'valid', 'test'])
  return Dataset(train_dataset, valid_dataset, test_dataset)


def add_average(lst, val, n):
  if len(lst) < n:
    lst.append(val)
  elif len(lst) == n:
    lst.pop(0)
    lst.append(val)
  elif len(lst) > n:
    raise Exception('List size is greater than n. This should never happen.')


def train(params):
  """Main trainining function."""
  if params.best_hp is not None:
    params = read_best_hp(params)

  class LeNetBase(tf.keras.Model):
    """LeNet for shared bottom."""

    def __init__(self):
      super(LeNetBase, self).__init__()
      self.conv1 = layers.Conv2D(
          filters=10,
          kernel_size=5,
          strides=(1, 1),
          padding='valid',
          activation='relu',
          input_shape=(36, 36, 1))
      self.conv2 = layers.Conv2D(
          filters=20,
          kernel_size=5,
          strides=(1, 1),
          padding='valid',
          activation='relu')
      self.fc1 = layers.Dense(
          params.size_sb, input_shape=(720,), activation='relu')
      self.fc2 = layers.Dense(
          params.size_sb, input_shape=(params.size_sb,), activation='relu')
      self.fc3 = layers.Dense(
          params.size_sb, input_shape=(params.size_sb,), activation='relu')
      self.fc4 = layers.Dense(
          params.size_sb, input_shape=(params.size_sb,), activation='relu')
      self.fc5 = layers.Dense(
          params.size_sb, input_shape=(params.size_sb,), activation='relu')
      self.fc6 = layers.Dense(
          params.size_sb, input_shape=(params.size_sb,), activation='relu')
      self.fc7 = layers.Dense(
          params.size_sb, input_shape=(params.size_sb,), activation='relu')

    def call(self, inputs, mask=None):
      x = inputs
      x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(self.conv1(x))
      x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(self.conv2(x))
      x = layers.Flatten()(x)
      # tune number of shared bottom layers
      if params.num_sb_layers >= 1:
        x = self.fc1(x)
      if params.num_sb_layers >= 2:
        x = self.fc2(x)
      if params.num_sb_layers >= 3:
        x = self.fc3(x)
      if params.num_sb_layers >= 4:
        x = self.fc4(x)
      if params.num_sb_layers >= 5:
        x = self.fc5(x)
      if params.num_sb_layers >= 6:
        x = self.fc6(x)
      if params.num_sb_layers >= 7:
        x = self.fc7(x)
      return x

  class LeNetTower(tf.keras.Model):
    """LeNet task specific tower."""

    def __init__(self):
      super(LeNetTower, self).__init__()
      self.fc0 = layers.Dense(
          params.size_ts, input_shape=(720,), activation='relu')
      self.fc1 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc2 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc3 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc4 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc5 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc6 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc7 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc8 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc9 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc = layers.Dense(10, input_shape=(params.size_ts,), activation=None)

    def call(self, inputs, mask=None):
      x = inputs
      if params.num_sb_layers == 0:  # change input shape
        x = self.fc0(x)
      if params.num_ts_layers >= 2:
        x = self.fc1(x)
      if params.num_ts_layers >= 3:
        x = self.fc2(x)
      if params.num_ts_layers >= 4:
        x = self.fc3(x)
      if params.num_ts_layers >= 5:
        x = self.fc4(x)
      if params.num_ts_layers >= 6:
        x = self.fc5(x)
      if params.num_ts_layers >= 7:
        x = self.fc6(x)
      if params.num_ts_layers >= 8:
        x = self.fc7(x)
      if params.num_ts_layers >= 9:
        x = self.fc8(x)
      if params.num_ts_layers >= 10:
        x = self.fc9(x)
      x = self.fc(x)
      return x

  class LeNetAuxNeckTower(tf.keras.Model):
    """LeNet auxiliary tower."""

    def __init__(self):
      super(LeNetAuxNeckTower, self).__init__()
      self.fc1 = layers.Dense(params.neck, input_shape=(720,), activation=None)
      self.fc2 = layers.Dense(10, input_shape=(params.neck,), activation=None)

    def call(self, inputs, mask=None):
      x = inputs
      x = self.fc1(x)
      x = self.fc2(x)
      return x

  class LeNetAuxTower(tf.keras.Model):

    def __init__(self):
      super(LeNetAuxTower, self).__init__()
      self.fc1 = layers.Dense(10, input_shape=(720,), activation=None)

    def call(self, inputs, mask=None):
      x = inputs
      x = self.fc1(x)
      return x

  class LeNetPTATower(tf.keras.Model):
    """LeNet PTA Tower."""

    def __init__(self):
      super(LeNetPTATower, self).__init__()
      self.fc0 = layers.Dense(
          params.size_ts, input_shape=(720,), activation='relu')
      self.fc1 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc2 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc3 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc4 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc5 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc6 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc7 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc8 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc9 = layers.Dense(
          params.size_ts, input_shape=(params.size_ts,), activation='relu')
      self.fc = layers.Dense(10, input_shape=(params.size_ts,), activation=None)

    def call(self, inputs, mask=None):
      x = inputs
      if params.num_sb_layers == 0:  # change input shape
        x = self.fc0(x)
      if params.num_ts_layers >= 2:
        x = self.fc1(x)
      if params.num_ts_layers >= 3:
        x = self.fc2(x)
      if params.num_ts_layers >= 4:
        x = self.fc3(x)
      if params.num_ts_layers >= 5:
        x = self.fc4(x)
      if params.num_ts_layers >= 6:
        x = self.fc5(x)
      if params.num_ts_layers >= 7:
        x = self.fc6(x)
      if params.num_ts_layers >= 8:
        x = self.fc7(x)
      if params.num_ts_layers >= 9:
        x = self.fc8(x)
      if params.num_ts_layers >= 10:
        x = self.fc9(x)
      x = self.fc(x)
      return x

  tf.logging.info(params)
  lebase = LeNetBase()
  ledigitr = LeNetTower()
  ledigitl = LeNetTower()
  ledigitr_aux_neck = LeNetAuxNeckTower()
  ledigitl_aux_neck = LeNetAuxNeckTower()
  ledigitr_aux = LeNetAuxTower()
  ledigitl_aux = LeNetAuxTower()
  ledigitr_pta1 = LeNetPTATower()
  ledigitr_pta2 = LeNetPTATower()
  ledigitr_pta3 = LeNetPTATower()
  ledigitr_pta4 = LeNetPTATower()
  ledigitl_pta1 = LeNetPTATower()
  ledigitl_pta2 = LeNetPTATower()
  ledigitl_pta3 = LeNetPTATower()
  ledigitl_pta4 = LeNetPTATower()
  ledigitr_aux_same = LeNetTower()
  ledigitl_aux_same = LeNetTower()

  dataset = load_dataset(params.batch_size, DATASET)
  if params.method == 'mgda':
    optimizer = tf.train.MomentumOptimizer(params.lr, momentum=0.9)
  else:
    optimizer = tf.keras.optimizers.SGD(params.lr, momentum=0.9)

  @tf.function()
  def train_step(trainx, labell, labelr):
    with tf.GradientTape(persistent=True) as tape:
      rep = lebase(trainx)
      outl = ledigitl(rep)
      outl_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=outl)
      outr = ledigitr(rep)
      outr_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=outr)

      l_loss = tf.reduce_mean(outl_loss)
      r_loss = tf.reduce_mean(outr_loss)
      loss = (1. - params.alpha) * l_loss + params.alpha * r_loss

      base_gradients = tape.gradient(loss, lebase.trainable_weights)
      l_digit_gradients = tape.gradient(loss, ledigitl.trainable_weights)
      r_digit_gradients = tape.gradient(loss, ledigitr.trainable_weights)

      optimizer.apply_gradients(zip(base_gradients, lebase.trainable_weights))
      optimizer.apply_gradients(
          zip(l_digit_gradients, ledigitl.trainable_weights))
      optimizer.apply_gradients(
          zip(r_digit_gradients, ledigitr.trainable_weights))
    return tf.reduce_sum(outl_loss), tf.reduce_sum(outr_loss)

  @tf.function()
  def train_aux_neck_step(trainx, labell, labelr):
    with tf.GradientTape(persistent=True) as tape:
      rep = lebase(trainx)
      outl = ledigitl(rep)
      outl_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=outl)
      outr = ledigitr(rep)
      outr_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=outr)

      # Auxiliary tasks
      outl_aux = ledigitl_aux_neck(rep) / (params.T_aux / 10.
                                          )  # tune temperature
      outl_loss_aux = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=outl_aux)
      outr_aux = ledigitr_aux_neck(rep) / (params.T_aux / 10.
                                          )  # tune temperature
      outr_loss_aux = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=outr_aux)

      # gamma controls strength of regularization
      l_loss = tf.reduce_mean(outl_loss) + 0.1 * params.gamma * tf.reduce_mean(
          outl_loss_aux)  # gamma in {0,1,2,3}
      r_loss = tf.reduce_mean(
          outr_loss) + 0.1 * params.gamma * tf.reduce_mean(outr_loss_aux)
      total_loss = (1 - params.alpha) * l_loss + params.alpha * r_loss

      base_gradients = tape.gradient(total_loss, lebase.trainable_weights)
      l_digit_gradients = tape.gradient(total_loss, ledigitl.trainable_weights)
      r_digit_gradients = tape.gradient(total_loss, ledigitr.trainable_weights)
      l_digit_aux_gradients = tape.gradient(total_loss,
                                            ledigitl_aux_neck.trainable_weights)
      r_digit_aux_gradients = tape.gradient(total_loss,
                                            ledigitr_aux_neck.trainable_weights)

      optimizer.apply_gradients(zip(base_gradients, lebase.trainable_weights))
      optimizer.apply_gradients(
          zip(l_digit_gradients, ledigitl.trainable_weights))
      optimizer.apply_gradients(
          zip(r_digit_gradients, ledigitr.trainable_weights))
      optimizer.apply_gradients(
          zip(l_digit_aux_gradients, ledigitl_aux_neck.trainable_weights))
      optimizer.apply_gradients(
          zip(r_digit_aux_gradients, ledigitr_aux_neck.trainable_weights))
    return tf.reduce_sum(outl_loss), tf.reduce_sum(outr_loss)

  @tf.function()
  def train_aux_step(trainx, labell, labelr):  # ablation study: without neck
    with tf.GradientTape(persistent=True) as tape:
      rep = lebase(trainx)
      outl = ledigitl(rep)
      outl_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=outl)
      outr = ledigitr(rep)
      outr_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=outr)

      # Auxiliary tasks
      outl_aux = ledigitl_aux(rep) / (params.T_aux / 10.)  # tune temperature
      outl_loss_aux = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=outl_aux)
      outr_aux = ledigitr_aux(rep) / (params.T_aux / 10.)  # tune temperature
      outr_loss_aux = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=outr_aux)

      # gamma controls strength of regularization
      l_loss = tf.reduce_mean(outl_loss) + 0.1 * params.gamma * tf.reduce_mean(
          outl_loss_aux)  # gamma in {0,1,2,3}
      r_loss = tf.reduce_mean(
          outr_loss) + 0.1 * params.gamma * tf.reduce_mean(outr_loss_aux)
      total_loss = (1 - params.alpha) * l_loss + params.alpha * r_loss

      base_gradients = tape.gradient(total_loss, lebase.trainable_weights)
      l_digit_gradients = tape.gradient(total_loss, ledigitl.trainable_weights)
      r_digit_gradients = tape.gradient(total_loss, ledigitr.trainable_weights)
      l_digit_aux_gradients = tape.gradient(total_loss,
                                            ledigitl_aux.trainable_weights)
      r_digit_aux_gradients = tape.gradient(total_loss,
                                            ledigitr_aux.trainable_weights)

      optimizer.apply_gradients(zip(base_gradients, lebase.trainable_weights))
      optimizer.apply_gradients(
          zip(l_digit_gradients, ledigitl.trainable_weights))
      optimizer.apply_gradients(
          zip(r_digit_gradients, ledigitr.trainable_weights))
      optimizer.apply_gradients(
          zip(l_digit_aux_gradients, ledigitl_aux.trainable_weights))
      optimizer.apply_gradients(
          zip(r_digit_aux_gradients, ledigitr_aux.trainable_weights))
    return tf.reduce_sum(outl_loss), tf.reduce_sum(outr_loss)

  @tf.function()
  def train_aux_same_step(
      trainx, labell,
      labelr):  # Ablation study: auxiliary tower same size as tower.
    with tf.GradientTape(persistent=True) as tape:
      rep = lebase(trainx)
      outl = ledigitl(rep)
      outl_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=outl)
      outr = ledigitr(rep)
      outr_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=outr)

      # Auxiliary tasks
      outl_aux = ledigitl_aux_same(rep) / (params.T_aux / 10.
                                          )  # tune temperature
      outl_loss_aux = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=outl_aux)
      outr_aux = ledigitr_aux_same(rep) / (params.T_aux / 10.
                                          )  # tune temperature
      outr_loss_aux = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=outr_aux)

      # gamma controls strength of regularization
      l_loss = tf.reduce_mean(outl_loss) + 0.1 * params.gamma * tf.reduce_mean(
          outl_loss_aux)  # gamma in {0,1,2,3}
      r_loss = tf.reduce_mean(
          outr_loss) + 0.1 * params.gamma * tf.reduce_mean(outr_loss_aux)
      total_loss = (1 - params.alpha) * l_loss + params.alpha * r_loss

      base_gradients = tape.gradient(total_loss, lebase.trainable_weights)
      l_digit_gradients = tape.gradient(total_loss, ledigitl.trainable_weights)
      r_digit_gradients = tape.gradient(total_loss, ledigitr.trainable_weights)
      l_digit_aux_gradients = tape.gradient(total_loss,
                                            ledigitl_aux.trainable_weights)
      r_digit_aux_gradients = tape.gradient(total_loss,
                                            ledigitr_aux.trainable_weights)

      optimizer.apply_gradients(zip(base_gradients, lebase.trainable_weights))
      optimizer.apply_gradients(
          zip(l_digit_gradients, ledigitl.trainable_weights))
      optimizer.apply_gradients(
          zip(r_digit_gradients, ledigitr.trainable_weights))
      optimizer.apply_gradients(
          zip(l_digit_aux_gradients, ledigitl_aux.trainable_weights))
      optimizer.apply_gradients(
          zip(r_digit_aux_gradients, ledigitr_aux.trainable_weights))
    return tf.reduce_sum(outl_loss), tf.reduce_sum(outr_loss)

  @tf.function()
  def train_pta_step(trainx, labell, labelr, num_pta):
    """Pseudo-Task Augmentation: https://arxiv.org/pdf/1803.04062.pdf."""
    with tf.GradientTape(persistent=True) as tape:
      rep = lebase(trainx)
      outl = ledigitl(rep)
      outl_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=outl)
      outr = ledigitr(rep)
      outr_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=outr)

      outl_loss_pta1 = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=ledigitl_pta1(rep))
      outr_loss_pta1 = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=ledigitr_pta1(rep))
      outl_loss_pta2 = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=ledigitl_pta2(rep))
      outr_loss_pta2 = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=ledigitr_pta2(rep))
      outl_loss_pta3 = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=ledigitl_pta3(rep))
      outr_loss_pta3 = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=ledigitr_pta3(rep))
      outl_loss_pta4 = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=ledigitl_pta4(rep))
      outr_loss_pta4 = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=ledigitr_pta4(rep))

      # PTA tasks
      if num_pta == 2:
        pta_loss_l = tf.reduce_mean(outl_loss_pta1) + tf.reduce_mean(
            outl_loss_pta2)
        pta_loss_r = tf.reduce_mean(outr_loss_pta1) + tf.reduce_mean(
            outr_loss_pta2)
      elif num_pta == 3:
        pta_loss_l = tf.reduce_mean(outl_loss_pta1) + tf.reduce_mean(
            outl_loss_pta2) + tf.reduce_mean(outl_loss_pta3)
        pta_loss_r = tf.reduce_mean(outr_loss_pta1) + tf.reduce_mean(
            outr_loss_pta2) + tf.reduce_mean(outr_loss_pta3)
      elif num_pta == 4:
        pta_loss_l = tf.reduce_mean(outl_loss_pta1) + tf.reduce_mean(
            outl_loss_pta2) + tf.reduce_mean(outl_loss_pta3) + tf.reduce_mean(
                outl_loss_pta4)
        pta_loss_r = tf.reduce_mean(outr_loss_pta1) + tf.reduce_mean(
            outr_loss_pta2) + tf.reduce_mean(outr_loss_pta3) + tf.reduce_mean(
                outr_loss_pta4)

      l_loss = tf.reduce_mean(outl_loss) + pta_loss_l
      r_loss = tf.reduce_mean(outr_loss) + pta_loss_r
      total_loss = (1 - params.alpha) * l_loss + params.alpha * r_loss

      base_gradients = tape.gradient(total_loss, lebase.trainable_weights)
      l_digit_gradients = tape.gradient(total_loss, ledigitl.trainable_weights)
      r_digit_gradients = tape.gradient(total_loss, ledigitr.trainable_weights)

      optimizer.apply_gradients(zip(base_gradients, lebase.trainable_weights))
      optimizer.apply_gradients(
          zip(l_digit_gradients, ledigitl.trainable_weights))
      optimizer.apply_gradients(
          zip(r_digit_gradients, ledigitr.trainable_weights))

    return tf.reduce_sum(outl_loss), tf.reduce_sum(outr_loss)

  @tf.function()
  def train_mgda_step(trainx, labell, labelr):
    """MGDA-UB algorithm proposed in: https://arxiv.org/abs/1810.04650."""
    rep = lebase(trainx)
    outl = ledigitl(rep)
    outl_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labell, logits=outl)
    outr = ledigitr(rep)
    outr_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labelr, logits=outr)

    l_loss = tf.reduce_mean(outl_loss)
    r_loss = tf.reduce_mean(outr_loss)

    # MGDA-UB
    l_digit_grad = optimizer.compute_gradients(l_loss, rep)[0][0]
    r_digit_grad = optimizer.compute_gradients(r_loss, rep)[0][0]
    if params.mgda_spec == 'reduce_mean':
      l_digit_grad = tf.reduce_mean(l_digit_grad, 0)
      r_digit_grad = tf.reduce_mean(r_digit_grad, 0)
    elif params.mgda_spec == 'reduce_sum':
      l_digit_grad = tf.reduce_sum(l_digit_grad, 0)
      r_digit_grad = tf.reduce_sum(r_digit_grad, 0)
    else:
      l_digit_grad = tf.reshape(l_digit_grad, [-1])
      r_digit_grad = tf.reshape(r_digit_grad, [-1])
    weight = tf.stop_gradient(
        tf.matmul(
            tf.reshape(l_digit_grad - r_digit_grad,
                       (1, -1)), tf.reshape(l_digit_grad, (-1, 1))) /
        tf.reduce_sum(tf.square(l_digit_grad - r_digit_grad)))
    weight = tf.clip_by_value(weight, 0.001, 0.999)  # value used in paper.

    new_loss = weight * l_loss + (1 - weight) * r_loss
    all_variables = (
        lebase.trainable_weights + ledigitl.trainable_weights +
        ledigitr.trainable_weights)
    gradvars = optimizer.compute_gradients(new_loss, all_variables)
    optimizer.apply_gradients(gradvars)
    return tf.reduce_sum(outl_loss), tf.reduce_sum(outr_loss)

  @tf.function()
  def train_uncertainty_step(trainx, labell, labelr):
    """Uncertainty-reweighting method: https://arxiv.org/pdf/1705.07115.pdf."""
    with tf.GradientTape(persistent=True) as tape:
      rep = lebase(trainx)
      outl = ledigitl(rep)
      outl_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labell, logits=outl)
      outr = ledigitr(rep)
      outr_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labelr, logits=outr)

      l_loss = tf.reduce_mean(outl_loss)

      global l_uncertainty
      global r_uncertainty
      if l_uncertainty is None:
        l_uncertainty = tf.Variable(1.0)
      if r_uncertainty is None:
        r_uncertainty = tf.Variable(1.0)

      l_clip_uncertainty = tf.clip_by_value(l_uncertainty, 0.01, 10.0)
      l_loss = l_loss / tf.exp(2 * l_clip_uncertainty) + l_clip_uncertainty
      r_loss = tf.reduce_mean(outr_loss)
      r_clip_uncertainty = tf.clip_by_value(r_uncertainty, 0.01, 10.0)
      r_loss = r_loss / tf.exp(2 * r_clip_uncertainty) + r_clip_uncertainty

      loss = l_loss + r_loss

      base_gradients = tape.gradient(loss, lebase.trainable_weights)
      l_digit_gradients = tape.gradient(loss, ledigitl.trainable_weights)
      r_digit_gradients = tape.gradient(loss, ledigitr.trainable_weights)
      uncertainty_gradients = tape.gradient(loss,
                                            [l_uncertainty, r_uncertainty])

      optimizer.apply_gradients(zip(base_gradients, lebase.trainable_weights))
      optimizer.apply_gradients(
          zip(l_digit_gradients, ledigitl.trainable_weights))
      optimizer.apply_gradients(
          zip(r_digit_gradients, ledigitr.trainable_weights))
      optimizer.apply_gradients(
          zip(uncertainty_gradients, [l_uncertainty, r_uncertainty]))
    return tf.reduce_sum(outl_loss), tf.reduce_sum(outr_loss)

  @tf.function()
  def eval_step(evalx, evalyl, evalyr):
    rep = lebase(evalx)
    outl = ledigitl(rep)
    outr = ledigitr(rep)
    l_loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(labels=evalyl, logits=outl))
    r_loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(labels=evalyr, logits=outr))

    l_pred = tf.math.argmax(outl, axis=1, output_type=tf.dtypes.int32)
    r_pred = tf.math.argmax(outr, axis=1, output_type=tf.dtypes.int32)
    l_acc = tf.math.count_nonzero(
        tf.equal(l_pred,
                 tf.math.argmax(evalyl, axis=1, output_type=tf.dtypes.int32)))
    r_acc = tf.math.count_nonzero(
        tf.equal(r_pred,
                 tf.math.argmax(evalyr, axis=1, output_type=tf.dtypes.int32)))
    Eval = collections.namedtuple('Eval',
                                  ['l_loss', 'r_loss', 'l_acc', 'r_acc'])
    return Eval(l_loss, r_loss, l_acc, r_acc)

  eval_metrics = {'l_loss': [], 'r_loss': [], 'l_acc': [], 'r_acc': []}
  for step in range(params.epoch):
    tf.logging.info('epoch: {}'.format(step))
    epoch_l_loss, epoch_r_loss = 0, 0
    for trainx, trainyl, trainyr in dataset.train:
      if params.method == 'mtl':
        batch_l_loss, batch_r_loss = train_step(trainx, trainyl, trainyr)
      elif params.method == 'aux_neck':
        batch_l_loss, batch_r_loss = train_aux_neck_step(
            trainx, trainyl, trainyr)
      elif params.method == 'aux':
        batch_l_loss, batch_r_loss = train_aux_step(trainx, trainyl, trainyr)
      elif params.method == 'aux_same':
        batch_l_loss, batch_r_loss = train_aux_same_step(
            trainx, trainyl, trainyr)
      elif params.method == 'pta2':
        batch_l_loss, batch_r_loss = train_pta_step(trainx, trainyl, trainyr, 2)
      elif params.method == 'pta3':
        batch_l_loss, batch_r_loss = train_pta_step(trainx, trainyl, trainyr, 3)
      elif params.method == 'pta4':
        batch_l_loss, batch_r_loss = train_pta_step(trainx, trainyl, trainyr, 4)
      elif params.method == 'mgda':
        batch_l_loss, batch_r_loss = train_mgda_step(trainx, trainyl, trainyr)
      elif params.method == 'uncertainty':
        batch_l_loss, batch_r_loss = train_uncertainty_step(
            trainx, trainyl, trainyr)
      else:
        raise Exception('Unknown method chosen.')
      epoch_l_loss += batch_l_loss / TRAIN_DATASET_SIZE
      epoch_r_loss += batch_r_loss / TRAIN_DATASET_SIZE
    tf.logging.info(
        'total train loss: {:.4f} || left digit loss: {:.4f} || right digit loss: {:.4f}'
        .format(((epoch_l_loss + epoch_r_loss)), epoch_l_loss, epoch_r_loss))

    if USE_VALID:
      eval_dataset = dataset.valid
    else:
      eval_dataset = dataset.test

    epoch_eval = {'l_loss': 0, 'r_loss': 0, 'l_acc': 0, 'r_acc': 0}
    for evalx, evalyl, evalyr in eval_dataset:
      eval_data = eval_step(evalx, evalyl, evalyr)
      for key, val in zip(eval_data._fields, eval_data):
        epoch_eval[key] += val
    for key in epoch_eval:
      epoch_eval[key] /= EVAL_DATASET_SIZE
      add_average(eval_metrics[key], epoch_eval[key], METRICS_AVERAGE)
      epoch_eval[key] = np.mean(eval_metrics[key])
    tf.logging.info('total eval loss: {:.4f}'.format(epoch_eval['l_loss'] +
                                                     epoch_eval['r_loss']))
    tf.logging.info(
        'left digit accuracy: {:.4f} || right digit accuracy: {:.4f}'.format(
            epoch_eval['l_acc'], epoch_eval['r_acc']))
  return epoch_eval


def hparams():
  return tf.HParams(
      best_hp='aux',
      method='mtl',
      lr=0.0001,
      alpha=0.5,
      beta=0.5,
      gamma=0.,
      batch_size=BATCH_SIZE,
      epoch=100,
      num_sb_layers=3,
      num_ts_layers=3,
      size_sb=50,
      size_ts=50,
      T_aux=10,
      neck=1,
      mgda_spec='default')


def main(argv):
  del argv  # Unused.

  master_hparams = hparams()

  tf.reset_default_graph()
  final_eval = train(master_hparams)
  tf.logging.info('final results: %s', final_eval)
  print('final results: %s', final_eval)
  return


if __name__ == '__main__':
  app.run(main)
