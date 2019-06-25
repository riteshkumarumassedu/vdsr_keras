#!/usr/bin/env python

from __future__ import print_function
from keras.models import Model
from keras.layers import Activation
from keras.layers import Conv2D, Input, add, subtract
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

Patch_width =256
Patch_height = 256

TARGET_IMG_SIZE = (512,512,1)

TRAIN_TEST_RATIO = (7, 3) # sum should be 10
BATCH_SIZE = 8
EPOCHS = 300
LR = 0.001
num_channels = 1



def tf_log10(x):
	numerator = tf.log(x)
	denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
	return numerator / denominator



# helper function to load the images

def load_input_img(filepath, patch):
	im = Image.open(filepath).convert('YCbCr')
	# im = np.array(im)
	# im = resize(im,(Patch_width, Patch_height), interpolation=INTER_CUBIC)

	if(patch==True):

		# inlcuding patch generation
		width, height = im.size  # Get dimensions

		left = (width - Patch_width) / 2
		top = (height - Patch_height) / 2
		right = (width + Patch_width) / 2
		bottom = (height + Patch_height) / 2

		im = im.crop((left, top, right, bottom))

		y, _, _ = im.split()
		y = np.array(y)
		y = np.reshape(y, (Patch_width, Patch_height))
	else:
		y, _, _ = im.split()
		y = np.array(y)
	return y

def load_target_img(filepath, patch):
	im = Image.open(filepath).convert('YCbCr')

	if(patch==True):

		# inlcuding patch generation
		width, height = im.size  # Get dimensions

		left = (width - Patch_width) / 2
		top = (height - Patch_height) / 2
		right = (width + Patch_width) / 2
		bottom = (height + Patch_height) / 2

		im = im.crop((left, top, right, bottom))

		y, _, _ = im.split()
		y = np.array(y)
		y = np.reshape(y, (Patch_width, Patch_height))
	else:
		y, _, _ = im.split()
		y = np.array(y)
	return y

# intial working version
# def load_img(filepath):
# 	img = Image.open(filepath).convert('YCbCr')
# 	y, _, _ = img.split()
# 	y = np.array(y)
# 	y = np.reshape(y, (1,y.shape[0], y.shape[1]))
# 	return y


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
	for t in target:
		x_file = os.path.join(DATA_X_PATH, t)
		x = load_input_img(x_file, patch=False)

		#print(x.shape)
		#x = resize(x, (Patch_height, Patch_width), interpolation=INTER_CUBIC)
		#x = resize(x, (TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1]), interpolation=INTER_CUBIC)

		#print(x.shape)
		x = np.reshape(x, (TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1],1))

		#print(x.shape)
		y_file = os.path.join(DATA_Y_PATH, t)
		y = load_target_img(y_file,patch=False)
		#y = resize(y, (TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1]), interpolation=INTER_CUBIC)

		y = np.reshape(y, ( TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1],1))

		# # normalize the data
		# x = x.astype('float32') / 255.0
		# y = y.astype('float32') / 255.0

		batch_x.append(x)
		batch_y.append(y)
	batch_x = np.array(batch_x)
	batch_y = np.array(batch_y)
	#print(batch_x.shape, batch_y.shape)
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


input_img = Input(shape=TARGET_IMG_SIZE)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model1 = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model1)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model2 = Activation('relu')(model)
model = add([model2, model1])

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model3 = Activation('relu')(model)
model = add([model3, model2])

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model4 = Activation('relu')(model)
model = add([model4, model3])

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model5 = Activation('relu')(model)
model = add([model5, model4])

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model6 = Activation('relu')(model)
model = add([model6, model5])

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

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)
print(model.shape)
res_img = model

#output_img = add([res_img, input_img])
output_img = res_img
model = Model(input_img, output_img)


adam = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001)
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
