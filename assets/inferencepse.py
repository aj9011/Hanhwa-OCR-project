# -*- coding:utf-8 -*-
import cv2
import time
import os
import numpy as np , pandas as pd
import tensorflow as tf
from tensorflow.python.client import timeline
from assets.utils.pse.utils import logger, cfg
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from assets.nets import psemodel as model
from assets.pse import pse

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

class InferencePSE:
    def __init__(self, parameters, file_urls):
        del_all_flags(tf.flags.FLAGS)
        tf.app.flags.DEFINE_string('f', '', 'kernel')
        tf.app.flags.DEFINE_string('gpu_list', parameters['gpu_list'], '')
        tf.app.flags.DEFINE_string('test_data_path', file_urls,'')
        tf.app.flags.DEFINE_string('checkpoint_path', parameters['checkpoint_path'], '')
        tf.app.flags.DEFINE_string('output_dir', file_urls.replace("original", "pse"), '')
        tf.app.flags.DEFINE_bool('no_write_images', parameters['no_write_images'], '')
        tf.app.flags.DEFINE_float('distance', parameters['distance'] , 'no big box cut off')
        tf.app.flags.DEFINE_float('seg_map_thresh', parameters['seg_map_thresh'] , 'segmentation threshold')
        tf.app.flags.DEFINE_integer('max_side_len', parameters['max_side_len'] , 'reshape image')
        tf.app.flags.DEFINE_integer('min_area_thresh', parameters['min_area_thresh'] , 'min area threshold we need small')
        
        self.FLAGS = tf.app.flags.FLAGS
        logger.setLevel(cfg.error)
  
    
    def get_images(self):
        '''
        find image files in test data path
        :return: list of files found
        '''
        files = []
        exts = ['jpg', 'png', 'jpeg', 'JPG', 'pgm', 'gif', 'tif']
        print(self.FLAGS.test_data_path)
        for parent, dirnames, filenames in os.walk(self.FLAGS.test_data_path):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        logger.info('Find {} images'.format(len(files)))
        return files


    def resize_image(self, im, max_side_len=1600):
        '''
        resize image to a size multiple of 32 which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        '''
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
        else:
            ratio = 1.

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
        logger.info('resize_w:{}, resize_h:{}'.format(resize_w, resize_h))
        im = cv2.resize(im, (int(resize_w), int(resize_h)))

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)


    def detect(self, seg_maps, timer, image_w, image_h, min_area_thresh=10, seg_map_thresh=0.9, ratio = 1):
        '''
        restore text boxes from score map and geo map
        :param seg_maps:
        :param timer:
        :param min_area_thresh:
        :param seg_map_thresh: threshhold for seg map
        :param ratio: compute each seg map thresh
        :return:
        '''
        if len(seg_maps.shape) == 4:
            seg_maps = seg_maps[0, :, :, ]
        #get kernals, sequence: 0->n, max -> min
        kernals = []
        one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
        zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)
        thresh = seg_map_thresh
        for i in range(seg_maps.shape[-1]-1, -1, -1):
            kernal = np.where(seg_maps[..., i]>thresh, one, zero)
            kernals.append(kernal)
            thresh = seg_map_thresh*ratio
        start = time.time()
        mask_res, label_values = pse(kernals, min_area_thresh)
        timer['pse'] = time.time()-start
        mask_res = np.array(mask_res)
        mask_res_resized = cv2.resize(mask_res, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        boxes = []
        for label_value in label_values:
            #(y,x)
            points = np.argwhere(mask_res_resized==label_value)
            points = points[:, (1,0)]
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            boxes.append(box)

        return np.array(boxes), kernals, timer

    
    def run(self):
        ret = 0
        try:
            os.makedirs(self.FLAGS.output_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise

        with tf.get_default_graph().as_default():
            input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            seg_maps_pred = model.model(input_images, is_training=False)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())
            config=tf.ConfigProto(log_device_placement=True)
            config.gpu_options.visible_device_list = self.FLAGS.gpu_list
            config.gpu_options.allow_growth = True
            config.allow_soft_placement=True
            with tf.Session(config=config) as sess:
                saver.restore(sess, self.FLAGS.checkpoint_path)
                print("restored!")
                sh_time1 = time.time()
                im_fn_list = self.get_images()
                print("num of images:",len(im_fn_list))
                for im_fn in im_fn_list:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                    logger.debug('image file:{}'.format(im_fn))

                    start_time = time.time()
                    im_resized, (ratio_h, ratio_w) = self.resize_image(im, max_side_len=self.FLAGS.max_side_len )
                    h, w, _ = im_resized.shape
                    timer = {'net': 0, 'pse': 0}
                    start = time.time()
                    seg_maps = sess.run(seg_maps_pred, feed_dict={input_images: [im_resized]})
                    timer['net'] = time.time() - start

                    boxes, kernels, timer = self.detect(seg_maps=seg_maps, timer=timer, 
                                                   image_w=w, image_h=h , 
                                                   min_area_thresh = self.FLAGS.min_area_thresh , 
                                                   seg_map_thresh= self.FLAGS.seg_map_thresh , 
                                                  )
                    logger.info('{} : net {:.0f}ms, pse {:.0f}ms'.format(
                        im_fn, timer['net']*1000, timer['pse']*1000))

                    if boxes is not None:
                        boxes = boxes.reshape((-1, 4, 2))
                        boxes[:, :, 0] /= ratio_w
                        boxes[:, :, 1] /= ratio_h
                        h, w, _ = im.shape
                        boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
                        boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)

                    duration = time.time() - start_time
                    logger.info('[timing] {}'.format(duration))

                    # save to file
                    print("num of boxes: ",len(boxes))
                    if boxes is not None:
                        res_file = os.path.join(
                            self.FLAGS.output_dir,
                            '{}.txt'.format(os.path.splitext(
                                os.path.basename(im_fn))[0]))


                        with open(res_file, 'w', encoding="utf-8-sig") as f:
                            num =0
                            point2 = []
                            point = []
                            for i in range(len(boxes)):
                                # to avoid submitting errors
                                box = boxes[i]
                                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5 :
                                    continue
                                check = np.array(box)[[0 , 2]]
                                check[...,:,0] = check[...,:,0]/w
                                check[...,:,1] = check[...,:,1]/h
                                endpoint = np.array([max(check[:,0]) , max(check[:,1])])
                                startpoint = np.array([min(check[:,0]) , min(check[:,1])])
                                distance= np.linalg.norm(startpoint- endpoint )
                                if distance > self.FLAGS.distance :
                                    continue
                                xx , yy = np.mean(check[:,0]) , np.mean(check[:,1])
                                point2.append([xx,yy])
                                point.append(box)
                                num += 1
                            pointt = np.array(point2)
                            total = pd.DataFrame(pointt)
                            metric = "euclidean"
                            dist_ex1 = cdist( total , total, metric=metric )
                            slice_ = pd.DataFrame(dist_ex1 < 0.1).drop_duplicates().index.tolist()
                            box2 = np.array(point)[slice_]
                            print(np.array(point).shape ,  "=>"  ,  box2.shape)
                            for box in box2 :
                                f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
                                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], 
                                              True, color=(255, 255, 0), thickness=2)
                    if not self.FLAGS.no_write_images: 
                        img_path = os.path.join(self.FLAGS.output_dir, os.path.basename(im_fn))
                        print("output img_path:",img_path)
                        cv2.imwrite(img_path, im[:, :, ::-1])

        return time.time() - sh_time1