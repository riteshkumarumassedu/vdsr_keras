#!/usr/bin/env python

from __future__ import print_function

import random

from keras.models import Model
from keras.layers import Activation
from keras.layers import Conv2D, Input, add, subtract, Flatten
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os, threading
import numpy as np


# laod config details

import yaml
from yaml import Loader
with open("config.yml", 'r') as config_file:
	config_params = yaml.load(config_file, Loader=Loader)


os.environ['CUDA_VISIBLE_DEVICES'] = config_params['cuda_device']

DATA_X_PATH = config_params['data_x_path']
DATA_Y_PATH = config_params['data_y_path']
patch_list = config_params['patch_list']


TRAIN_TEST_RATIO = config_params['train_test_ratio']
BATCH_SIZE = config_params['batch_size']
EPOCHS = config_params['epochs']
LR = config_params['LR']


def tf_log10(x):
	numerator = tf.log(x)
	denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
	return numerator / denominator

def normalize_data(raw_data):
	norm_data = config_params['max_pixel_val']*(raw_data - np.min(raw_data)) / np.ptp(raw_data)
	return norm_data


def generate_numpy_slices():

	np_files = {}
	# get all the numpy data and generate the slices
	np_files['data_x_path'] = os.listdir(config_params['data_x_path'])
	np_files['data_y_path'] = os.listdir(config_params['data_y_path'])
	np_files['data_test_path'] = os.listdir(config_params['data_test_path'])

	for one_dataset in np_files:
		print(" For: " + one_dataset)

		# if there are no numpy slices in the folder, then only generate the slices

		if len(np_files[one_dataset]) < 4:
			print(" Generating the numpy slices for: " + one_dataset)

			#  get all the numpy files under the specific dir

			if 'x' in one_dataset:
				all_np_files_in_one_set = os.listdir(config_params['numpy_x_path'])
				np_path = config_params['numpy_x_path']
				data_path = config_params['data_x_path']

			elif 'y' in one_dataset:
				all_np_files_in_one_set = os.listdir(config_params['numpy_y_path'])
				np_path = config_params['numpy_y_path']
				data_path = config_params['data_y_path']

			elif 'test' in one_dataset:
				all_np_files_in_one_set = os.listdir(config_params['numpy_test_path'])
				np_path = config_params['numpy_test_path']
				data_path = config_params['data_test_path']

			# print(all_np_files_in_one_set)

			for one_file in all_np_files_in_one_set:
				if '.npy' in one_file:
					file_name = one_file.split('.')[1]

					# load the numpy data for that file
					np_data = np.load(np_path + one_file)

					# print(np_data.shape)
					# get number of slices
					num_slices = np_data.shape[2]

					for slice_num in range(num_slices):
						# print(file_name, slice_num, np_data[:,:,slice_num].shape)
						np.save( file=data_path+'/'+file_name + '_' + str(slice_num) + '.npy', arr=np_data[:,:,slice_num])

	print("NumPy slices Ganaration : Done !!")
	return



"""
	Helper function to generate data patches 
	for both input and target numPy data 
"""
def generate_patch(input_pixels, target_pixels, patch_size):

	w, h = input_pixels.shape
	th, tw = patch_size
	if w == tw and h == th:

		input_pixels = input_pixels[0:w, 0:h]
		target_pixels = target_pixels[0:w, 0:h]
		return input_pixels, target_pixels

	i = random.randint(0, h - th)
	j = random.randint(0, w - tw)

	input_pixels = input_pixels[i:i+th, j:j+tw]
	target_pixels = target_pixels[i:i+th, j:j+tw]

	return input_pixels, target_pixels


def load_numpy_data(in_img, tgt_img, patch_size):

	"""	Laod both  input and target images """
	in_img = np.load(in_img)
	tgt_img = np.load(tgt_img)

	"""	
		Generate desired patches 
		Send both "input" and "target" image together 
			so that the patches are spatially same
	"""
	inp, tgt = generate_patch(in_img, tgt_img, patch_size)

	return inp, tgt


# only NumPy files are supported
def get_data_list(data_path):

	l = os.listdir(data_path)
	train_list = []
	for f in l:
		if f[-4:] == '.npy':
			train_list.append(f)
	return train_list


def get_data_batch(target_list, offset):
	target = target_list[offset:offset+BATCH_SIZE]
	batch_x = []
	batch_y = []

	"""  select random patch size """
	idx = random.randint(1, 3)
	patch_size = patch_list[idx]

	"""	Ganrate the image batch	"""

	for t in target:
		x_data = os.path.join(DATA_X_PATH, t)
		y_data = os.path.join(DATA_Y_PATH, t)

		x, y = load_numpy_data(x_data, y_data, patch_size=patch_size)

		x = normalize_data(x)
		y = normalize_data(y)


		# x = np.reshape(x, (x.shape[0], x.shape[1], 1))
		# y = np.reshape(y, (y.shape[0], y.shape[1], 1))

		x = np.reshape(x, (1, x.shape[0], x.shape[1] ))
		y = np.reshape(y, (1, y.shape[0], y.shape[1] ))

		batch_x.append(x)
		batch_y.append(y)
	batch_x = np.array(batch_x)
	batch_y = np.array(batch_y)
	return batch_x, batch_y


class threadsafe_iter:

	"""Takes an iterator/generator and makes it thread-safe by
		serializing call to the `next` method of given iterator/generator.
	"""

	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return self.it.next()


def data_gen(target_list):
	while True:
		for step in range(len(target_list)//BATCH_SIZE):
			offset = step*BATCH_SIZE
			batch_x, batch_y = get_data_batch(target_list, offset)
			yield (batch_x, batch_y)


# method to compute the PSNR values
def PSNR(y_true, y_pred):
	psnr = tf.image.psnr(y_true, y_pred, max_val=config_params['max_pixel_val'])
	return psnr


# method to compute the Similarity index

def SSIM(y_true, y_pred):
	return tf.image.ssim(y_pred, y_true, config_params['max_pixel_val'])


""" generate the numpy 2D slices form the numpy volumes """
generate_numpy_slices()

"""	Get the training and testing data """

img_list = get_data_list(DATA_X_PATH)
imgs_to_train = (len(img_list) * TRAIN_TEST_RATIO[0] )// 10

train_list = img_list[:imgs_to_train]
val_list = img_list[imgs_to_train:]



input_img = Input(shape=(1, None, None))

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(input_img)
model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(model)
# model1 = Activation('relu')(model)
#
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(model1)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(model)
# model2 = Activation('relu')(model1)
#
# model = add([model2, model1])
#
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(model)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(model)
# model3 = Activation('relu')(model)
#
# model = add([model3, model2])
#
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(model)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(model)
# model4 = Activation('relu')(model)
# model = add([model4, model3])


# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(model)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(model)
# model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(model)
model = Activation('relu')(model)
model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(model)
res_img = model

output_img = add([res_img, input_img])
#output_img = res_img
model = Model(input_img, output_img)


adam = Adam(lr=LR)
# sgd = SGD(lr=1e-5, momentum=0.9, decay=1e-5, nesterov=False)
model.compile(adam, loss='mse', metrics=[PSNR, "accuracy"])
model.summary()

with open('./model/vdsr_architecture.json', 'w') as f:
	f.write(model.to_json())


filepath="./checkpoints/model-{epoch:02d}-{PSNR:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor=PSNR, verbose=1, mode='max',period=config_params['save_every'])

lr_scheduler = ReduceLROnPlateau(monitor='val_PSNR', factor=0.5, patience=10, min_lr=0.00001)

callbacks_list = [checkpoint,lr_scheduler]


model.fit_generator(data_gen(train_list), steps_per_epoch=config_params['steps_per_epoch'], \
					validation_data=data_gen(val_list), validation_steps=len(val_list) // BATCH_SIZE, \
					epochs=EPOCHS, use_multiprocessing=True, workers=1, callbacks=callbacks_list)


model.save('./model/vdsr_model.h5')  # creates a HDF5 file

del model  # deletes the existing model
