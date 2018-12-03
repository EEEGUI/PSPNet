from __future__ import print_function
import cv2
from scipy import misc

from model import PSPNet101, PSPNet50
from tools import *


class VideoSegmentation:
    def __init__(self, video_dir):
        self.cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'model': PSPNet101}
        self.snapshot_dir = './save-model'
        self.flipped_eval = False
        filename = video_dir.split('/')[-1].split('.')[0]
        self.cap = cv2.VideoCapture(video_dir)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.out = cv2.VideoWriter('output/seg_%s.avi' % filename, fourcc, 15.0, (int(self.h), int(self.w)))
        self._load_model()

    def _load_model(self):
        param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'model': PSPNet101}
        crop_size = param['crop_size']
        num_classes = param['num_classes']
        PSPNet = param['model']

        # preprocess images
        self.img = tf.placeholder(dtype=tf.float32, shape=[self.h, self.w, 3], name='input')
        img_shape = tf.shape(self.img)
        h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))
        self.img = preprocess(self.img, h, w)

        # Create network.
        net = PSPNet({'data': self.img}, is_training=False, num_classes=num_classes)
        with tf.variable_scope('', reuse=True):
            flipped_img = tf.image.flip_left_right(tf.squeeze(self.img))
            flipped_img = tf.expand_dims(flipped_img, dim=0)
            net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=num_classes)

        raw_output = net.layers['conv6']

        # Do flipped eval or not
        if self.flipped_eval:
            flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
            flipped_output = tf.expand_dims(flipped_output, dim=0)
            raw_output = tf.add_n([raw_output, flipped_output])

        # Predictions.
        raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
        raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        self.pred = decode_labels(raw_output_up, img_shape, num_classes)

        # Init tf Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()

        self.sess.run(init)

        restore_var = tf.global_variables()

        ckpt = tf.train.get_checkpoint_state(self.snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=restore_var)
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            loader.restore(self.sess, ckpt.model_checkpoint_path)
            print("Restored model parameters from {}".format(ckpt.model_checkpoint_path))
        else:
            print('No checkpoint file found.')

    def process(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is True:
                img = frame[:, :, ::-1]
                preds = self.sess.run(self.pred, feed_dict={self.img: img})
                self.out.write(preds)
                # cv2.imshow('frame1',frame1)
                # cv2.imshow('frame2', frame2)
                # cv2.waitKey(1)
                # if cv2.waitKey(1) & 0xFF == 27:
                #     break
            else:
                break
            # Release everything if job is finished
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    for file in os.listdir(r'input'):
        video_dir = 'input/' + file
        vs = VideoSegmentation