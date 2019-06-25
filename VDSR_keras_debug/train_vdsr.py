#!/usr/bin/env python

from __future__ import print_function

import random

from keras.models import Model
from keras.layers import Activation
from keras.layers import Conv2D, Input, add, subtract, Flatten
import tensorflow as tf
from keras.optimizers import  Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from cv2 import resize, imread, INTER_CUBIC
import os, threading
import numpy as np
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

DATA_X_PATH = "./data/train_x/"
DATA_Y_PATH = "./data/train_y/"

# patch_size = (32,32)

#patch_list = [(32, 32), (64, 64), (128, 128), (256, 256)]
patch_list = [(512,512),(512,512),(512,512),(512,512)]
TARGET_IMG_SIZE = (512,512,1)

TRAIN_TEST_RATIO = (7, 3) # sum should be 10
BATCH_SIZE = 8
EPOCHS = 700
LR = 0.001
num_channels = 1



def tf_log10(x):
	numerator = tf.log(x)
	denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
	return numerator / denominator


# Helper function to generate image patches for both input and target image

def generate_patch(input_image,target_image, patch_size):
	w, h = input_image.size
	th, tw = patch_size
	if w == tw and h == th:

		input_image = input_image.crop((0, 0, 0 + w, 0 + h))
		target_image = target_image.crop((0, 0, 0 + w, 0 + h))
		return input_image, target_image

	i = random.randint(0, h - th)
	j = random.randint(0, w - tw)

	input_image = input_image.crop((j, i, j + tw, i + th))
	target_image = target_image.crop((j, i, j + tw, i + th))

	return input_image, target_image


def load_image_data(in_img, tgt_img, patch_size):

	# laod both of the images
	in_img = Image.open(in_img).convert('YCbCr')
	tgt_img = Image.open(tgt_img).convert('YCbCr')

	# get only the intensity channel

	in_img, _, _ = in_img.split()
	tgt_img, _, _ = tgt_img.split()

	# generate desired patches
	inp, tgt = generate_patch(in_img, tgt_img, patch_size)

	return inp, tgt


# supporting only JPG and PNG
def get_image_list(data_path):
	l = os.listdir(data_path)
	train_list = []
	for f in l:
		if f[-4:] == '.jpg'or f[-4:] == '.png':
			train_list.append(f)
	return train_list


def get_image_batch(target_list, offset):
	target = target_list[offset:offset+BATCH_SIZE]
	batch_x = []
	batch_y = []
	idx = random.randint(1, 3)
	patch_size = patch_list[idx]
	Patch_height, Patch_width = patch_size

	for t in target:

		# Patch_height, Patch_width = patch_size
		# patch_size = (64,64)
		# print("ewr", patch_size)
		# get input and target image
		x_file = os.path.join(DATA_X_PATH, t)
		y_file = os.path.join(DATA_Y_PATH, t)

		x,y = load_image_data(x_file,y_file, patch_size=patch_size)


		x_patch, y_patch  = np.array(x), np.array(y)
		# print(x_patch.shape,y_patch.shape)
		#x = resize(x, (Patch_height, Patch_width), interpolation=INTER_CUBIC)
		#x = resize(x_patch, (TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1]), interpolation=INTER_CUBIC)
		#x = np.reshape(x, (TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1],1))
		x = np.reshape(x_patch, (x_patch.shape[0],x_patch.shape[1],1))


		#y = resize(y_patch, (TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1]), interpolation=INTER_CUBIC)
		#y = np.reshape(y, ( TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1],1))
		y = np.reshape(y_patch, (y_patch.shape[0], y_patch.shape[1], 1))


		# x = y.copy()
		# x = resize(x, (128,128))
		# x = resize(x, (Patch_height, Patch_width), interpolation=INTER_CUBIC)
		# x = np.reshape(x, (x.shape[0], x.shape[1], 1))

		# print(x.shape, y.shape)
		# print(x.shape,y.shape)
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



def image_gen(target_list):
	while True:
		for step in range(len(target_list)//BATCH_SIZE):
			offset = step*BATCH_SIZE
			batch_x, batch_y = get_image_batch(target_list, offset)
			yield (batch_x, batch_y)


# method to compute the PSNR values
def PSNR(y_true, y_pred):
	psnr = tf.image.psnr(y_true, y_pred, max_val=255.0)
	return psnr


# method to compute the Similarity index

def SSIM(y_true, y_pred):
	max_pixel = 255.0
	return tf.image.ssim(y_pred, y_true, max_pixel)
 


# Get the training and testing data
img_list = get_image_list(DATA_X_PATH)

imgs_to_train = len(img_list) *  TRAIN_TEST_RATIO[0] // 10

train_list = img_list[:imgs_to_train]
val_list = img_list[imgs_to_train:]


input_img = Input(shape=(None, None,1))

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(input_img)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
model = Activation('relu')(model)
# model = add([model2, model1])
#
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# model = add([model3, model2])
#
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# # model = add([model4, model3])
#
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# # model = add([model5, model4])
#
#
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# model = add([model6, model5])

# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# #model = add([model3, model2])
#
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
#model = add([model4, model3])

#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model5 = Activation('relu')(model)
#model = add([model5, model4])

#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model4 = Activation('relu')(model)
#model = add([model4, model3])

#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model5 = Activation('relu')(model)
#model = add([model5, model4])

#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model6 = Activation('relu')(model)
#model = add([model6, model5])

#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model7 = Activation('relu')(model)
#model = add([model7, model6])

#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model = Activation('relu')(model)
#model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#model8 = Activation('relu')(model)

#model = add([model8, model7])

# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
# model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
# model = Activation('relu')(model)
model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_last')(model)
print(model.shape)
res_img = model

# output_img = add([res_img, input_img])
output_img = res_img
model = Model(input_img, output_img)


adam = Adam(lr=LR)
# sgd = SGD(lr=1e-5, momentum=0.9, decay=1e-5, nesterov=False)
model.compile(adam, loss='mse', metrics=[PSNR, "accuracy"])
model.summary()

with open('./model/vdsr_architecture.json', 'w') as f:
	f.write(model.to_json())


filepath="./checkpoints/model-{epoch:02d}-{PSNR:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor=PSNR, verbose=1, mode='max',period=5)

lr_scheduler = ReduceLROnPlateau(monitor='val_PSNR', factor=0.5, patience=10, min_lr=0.00001)

callbacks_list = [checkpoint,lr_scheduler]


model.fit_generator(image_gen(train_list), steps_per_epoch=len(train_list) // BATCH_SIZE, \
					validation_data=image_gen(val_list), validation_steps=len(val_list) // BATCH_SIZE, \
					epochs=EPOCHS, workers=1, callbacks=callbacks_list)


model.save('./model/vdsr_model.h5')  # creates a HDF5 file

del model  # deletes the existing model
