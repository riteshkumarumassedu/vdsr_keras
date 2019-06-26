import sys
import os
from keras.models import model_from_json
import numpy as np

import yaml
from yaml import Loader

with open("config.yml", 'r') as config_file:
    config_params = yaml.load(config_file, Loader=Loader)

def num_char(x):

    tmp = (x.split('_')[1]).split('.')[0]
    return int(tmp)



def stitch_numpy_slices():
    np_slices_path = config_params['np_slices_path']

    output_path = './numpy_stitch_output/'

    np_slices = os.listdir(np_slices_path)

    # remove non-numpy things from the list
    for one_file in np_slices:
        if '.npy' not in one_file:
            np_slices.remove(one_file)

    np_slices.sort(key=num_char)
    print(np_slices)
    # stitch same files

    file_basename = ''
    output = None
    for one_file in np_slices:
        print(one_file)
        if file_basename == '':
            file_basename = one_file.split('_')[0]
            print(file_basename)
            # get the numpy data from the file
            output = np.load(np_slices_path+one_file)
            output = np.reshape(output, (output.shape[0], output.shape[1], 1))

        else:
            if file_basename in one_file:
                tmp = np.load(np_slices_path + one_file)
                tmp = np.reshape(tmp, (tmp.shape[0], tmp.shape[1], 1))
                output = np.append(output, tmp, axis=2)
            else:

                # first save the previous file and reset the file_basename and output
                print("Saving the stitched file : " + str(output.shape))
                np.save(file=output_path+file_basename+".npy", arr=output)
                output = np.empty()

                file_basename = one_file.split('_')[0]

                # get the numpy data from the file
                tmp = np.load(np_slices_path + one_file)
                tmp = np.reshape(tmp, (tmp.shape[0], tmp.shape[1], 1))
                output = np.append(output, tmp, axis=2)

    if(file_basename != ''):
        print("Saving the stitched file : " + str(output.shape))
        np.save(file=output_path + file_basename + ".npy", arr=output)




if __name__ == '__main__':

    """ generates the np_stitched file """
    stitch_numpy_slices()
    exit()

    json_path = config_params['json_path']
    w_path = sys.argv[1]
    img_path = config_params['test_images_path']
    dst_path = config_params['test_dest_path']
    target_size = config_params['pred_tgt_size']

    print("--------------------------------")
    print('json_path : ', json_path)
    print('w_path : ', w_path)
    print('img_path : ', img_path)
    print('dst_path : ', dst_path)
    print("--------------------------------")

    with open(json_path, 'r') as f:
        vdsr = model_from_json(f.read())
    vdsr.load_weights("checkpoints/" + w_path)

    li = os.listdir(img_path)

    target_path = '%s/%s/' % (img_path, dst_path)
    os.makedirs(target_path, exist_ok=True)

    for filename in li:
        if filename[-4:] == '.npy':
            print("executing :", filename)
            np_file = os.path.join(img_path, filename)
            inp_pixels = np.load(np_file)
            inp_pixels = np.reshape(inp_pixels, (1, 1, inp_pixels.shape[0], inp_pixels.shape[1]))

            out_pixels = vdsr.predict(inp_pixels)

            out_pixels = np.reshape(out_pixels, target_size)

            np.save(file=target_path + "/" + filename, arr=out_pixels)
