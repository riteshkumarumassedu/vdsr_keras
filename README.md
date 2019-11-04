# vdsr_keras
Implementation of CVPR2016 Paper: "Accurate Image Super-Resolution Using Very Deep Convolutional Networks"(http://cv.snu.ac.kr/research/VDSR/)

Multiple version of VDSR implementation using Keras for dicom images, for numpy data, for jpeg images
Deep Learning driven model for Single Image Super Resolution (SISR) to construct high resolution images from lower resolution images


## How to Run the Training:

1.	Set all the necessary parameters in the config.yml file such as batch size, upscale factor, learning rate etc.
2.	 Run the following command from the terminal:
Python main.py –model < either srcnn or vdsr>

All the trained model weights would be saved in the root directory of this project.

## How to Run the Inference:

1.	Set all the necessary parameters in the config.yml file 
2.	Run the following command from the terminal:
Python test.py –input <image to test> --ouptut <output image name>  --model  < model weights file to use for inference>


## Directory Tree:
<pre>
.VDSR_keras_numpy_dicom   Code Root Directory<br>
|- data                   Images directory
  |- dicom_to_numpy       dicom files directory
    |- test               test dicom files
    |- x                  train (input) dicom files
    |- y                  train (output/label) dicom files
  |- test                 test NumPy data extracted from dicom files 
  |- train_x              train(input) NumPy data
  |- train_y              train(output) NumPy data
|- model                  directory to save model architecture
|- results                inference output
|- dicom_smaple_path      dicom sample file to create dicom output files
|- checkpoints            directory to save trained models weights
|- Config.yml             config file for the model
|- train_vdsr.py          VDSR model implementation for training
|- predict_vdsr.py        VDSR model inference code
</pre>
