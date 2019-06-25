import sys
import os
from keras.models import model_from_json
import numpy as np
from cv2 import imread, resize, INTER_CUBIC
import matplotlib.pylab as plt
from PIL import Image

if __name__ == '__main__':
    json_path = sys.argv[1]
    w_path = sys.argv[2]
    img_path = sys.argv[3]
    dst_path = sys.argv[4]
    target_size = (512,512) # should be changed according to size of output image
    print("--------------------------------")
    print('json_path : ', json_path)
    print('w_path : ', w_path)
    print('img_path : ', img_path)
    print('dst_path : ', dst_path)
    print("--------------------------------")

    with open(json_path, 'r') as f:
        vdsr = model_from_json(f.read())
    vdsr.load_weights(w_path)

    li = os.listdir(img_path)

    target_path = '%s/%s/' % (img_path, dst_path)
    os.makedirs(target_path, exist_ok=True)

    for filename in li:
        if filename[-4:] == '.jpg' or filename[-4:] == '.png':

            input_file=os.path.join(img_path, filename)
            img = Image.open(input_file).convert('YCbCr')
            y, cb, cr = img.split()
            y = np.array(y)
            y = np.reshape(y, (1,y.shape[0], y.shape[1],1))
            img = vdsr.predict(y)

            out_img_y = img
            #print(out_img_y.max(), out_img_y.min())
            #print(out_img_y)
            #exit()
            #out_img_y *= 255.0
            out_img_y = np.reshape(out_img_y,target_size)
            out_img_y = 255*(out_img_y - np.min(out_img_y))/np.ptp(out_img_y)
            out_img_y = out_img_y.clip(0, 255.0)
            out_img_y = Image.fromarray(np.uint8(out_img_y), mode='L')

            out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
            out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
            out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

            out_img.save(target_path + "/" + filename)

            # img = imread(os.path.join(img_path, filename))
            # img = resize(img, dsize=target_size, interpolation=INTER_CUBIC)
            # img.astype('float32') / 255.0
            # img = img.reshape((1,) + target_size + (3,))
            # img = vdsr.predict(img)
            # img = img.reshape(target_size + (3,))
            # print(img)
            # # img = Image.fromarray(img)
            # # plt.imshow(img, cmap='gray')
            # plt.imsave(fname=target_path + "/" + filename, arr=img, cmap='bone')
            # # toimage(img, cmin=0.0, cmax=255, mode='RGB').save('%s/%s' % (target_path, filename))
        else:
            pass


    # for filename in li:
    #     if filename[-4:] == '.jpg' or filename[-4:] == '.png':
    #         img = imread(os.path.join(img_path, filename))
    #         img = resize(img, dsize=target_size, interpolation=INTER_CUBIC)
    #         img = np.array(img) / 255.0 - 1.
    #         img = img.reshape((1,)+target_size+(3,))
    #         img = vdsr.predict(img)
    #         print(filename)
    #         img = img.reshape(target_size+(3,))
    #         img = (0.5 * img + 0.5) * 255
    #         img = Image.fromarray(img)
    #         # img.save(target_path+"/"+filename)
    #         plt.imsave(fname=target_path+"/"+filename, arr= img, cmap='bone')
    #         # toimage(img, cmin=0.0, cmax=255, mode='RGB').save('%s/%s' % (target_path, filename))
    #     else:
    #         pass
