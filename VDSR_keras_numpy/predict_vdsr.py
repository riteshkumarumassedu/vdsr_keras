import sys
import os
from keras.models import model_from_json
import numpy as np

# import train_vdsr


import yaml
from yaml import Loader

with open("config.yml", 'r') as config_file:
    config_params = yaml.load(config_file, Loader=Loader)

def num_char(x):

    tmp = (x.split('_')[1]).split('.')[0]
    return int(tmp)



def stitch_numpy_slices():
    np_slices_path = config_params['np_slices_path']
    output_path = config_params['np_stitched_output_path']

    """ Get all the files under the results directory"""
    np_slices = os.listdir(np_slices_path)

    """
         remove non-numpy things from the list
         segregate slices for the same chunk 
    """

    seg_files = {}
    for one_file in np_slices:

        if '.npy' in one_file:

            """ Get the file basename"""
            file_basename = one_file.split('_')[0]
            file_to_add = np_slices_path + one_file

            if file_basename not in seg_files:

                """ If not seg file dict """
                tmp = list()
                tmp.append(file_to_add)
                seg_files[file_basename] = tmp

            else:

                """ if already in seg_files list"""
                tmp = seg_files[file_basename]
                tmp.append(file_to_add)
                seg_files[file_basename] = tmp


    for one_base_file in seg_files.keys():
        print(" working on :" + one_base_file)
        base_name = one_base_file
        all_slices = seg_files[one_base_file]

        print("Total slices :" , len(all_slices))
        """ sort all the slices in place """
        all_slices.sort(key=num_char)

        output = None
        for one_slice in all_slices:
            if output is None:
                output = np.load(one_slice)
                output = np.reshape(output, (output.shape[0], output.shape[1], 1))
            else:
                tmp = np.load(one_slice)
                tmp = np.reshape(tmp, (tmp.shape[0], tmp.shape[1], 1))
                output = np.append(output, tmp, axis=2)

        print(" Saving the stitched file with shape: " + str(output.shape))
        np.save(file= output_path + str(base_name) + ".npy", arr=output)

    return



if __name__ == '__main__':

    run_test = False

    if run_test:

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

                """  Load numpy file  and reshape """
                inp_pixels = np.load(np_file)
                inp_pixels = np.reshape(inp_pixels, (1, 1, inp_pixels.shape[0], inp_pixels.shape[1]))

                max_scale = inp_pixels.max()
                """ Get the output from the model and reshape"""
                out_pixels = vdsr.predict(inp_pixels)

                out_pixels = np.reshape(out_pixels, target_size)

                # normalize the pixels based on the max pixel value of the input image
                # out_pixels = max_scale*(out_pixels - np.min(out_pixels)) / np.ptp(out_pixels)
                out_pixels = np.clip(a=out_pixels,a_min=0, a_max=out_pixels.max())
                np.save(file=target_path + "/" + filename, arr=out_pixels)

    else:
        print(" ++ Not Running the test segment. ++ ")

    """ generates the np_stitched file """

    stitch_numpy_slices()