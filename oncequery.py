#coding=utf-8
import tensorflow as tf
import alexnet
import io
import json
import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy import misc

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', 'log',
                    """Model checkpoint dir path.""")

tf.app.flags.DEFINE_string('feature_lib', 'output/fc2_features.json',
                    """Model checkpoint dir path.""")

IMG_SIZE=227
MOVING_AVERAGE_DECAY = 0.9999


def query(filepath,output,rank):
    #提取待搜索图片的特征
    qfeature=getQfeature(filepath)
    featurelib=getFeaturelib()
    dists=[]

    for image_feature in list(featurelib.values()):
        dist=np.linalg.norm(np.array(qfeature)-np.array(image_feature))
        dists.append(dist)

    distlist=list(zip(list(featurelib.keys()),dists))
    top_k=heapq.nsmallest(rank,distlist, key= lambda d:d[1])
    res=list(zip(*top_k))[0]

    with open(output, 'w') as sf:
        sf.write(json.dumps(res))
        print('Write result ok!')

def getFeaturelib():
    featurelib=FLAGS.feature_lib
    with open(featurelib,'r') as f:
        featurelib_dict=json.loads(f.read())
        return featurelib_dict

def getQfeature(filepath):
    with tf.Graph().as_default() as g:
        image=read(filepath)
        logits = alexnet.inference(image)
        fc1 = g.get_tensor_by_name("fc1/fc1:0")
        fc2 = g.get_tensor_by_name("fc2/fc2:0")

        fc1_norm = tf.nn.l2_normalize(fc1, 1)
        fc2_norm = tf.nn.l2_normalize(fc2, 1)
        fc3_norm = tf.nn.l2_normalize(logits, 1)

        EMA = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = EMA.variables_to_restore()
        # for name in variables_to_restore:
        # 	print(name)
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:

        #tf.train.get_checkpoint_state函数通过checkpoint文件找到模型文件�?
        #返回值有model_checkpoint_path和all_model_checkpoint_paths两个属性�?
        # 其中model_checkpoint_path保存了最新的tensorflow模型文件的文件名�?
        # all_model_checkpoint_paths则有未被删除的所有tensorflow模型文件的文件名�?
            init = tf.global_variables_initializer()
            sess.run(init)

            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            fc1_list=[];fc2_list=[];fc3_list=[]

            _,_fc1,_fc2,_fc3=sess.run([image,fc1_norm,fc2_norm,fc3_norm])
            put_2darray(_fc1,fc1_list)
            put_2darray(_fc2,fc2_list)
            put_2darray(_fc3,fc3_list)

            return fc2_list


def read(filepath):
    I = misc.imread(filepath)
    distort_img=tf.cast(I,tf.float32)
    distorted_image = tf.image.resize_images(distort_img, [IMG_SIZE, IMG_SIZE])
    image=tf.image.per_image_standardization(distorted_image)
    img=tf.reshape(image,[1,IMG_SIZE,IMG_SIZE,3])
    #print(img)
    return img


def put_2darray(_2darray,li):
    _li=_2darray.tolist()
    for line in _li:
        li.append(line)


def save(feature_list,file_nm):
    with io.open(file_nm,'w',encoding='utf-8') as file:
        file.write(json.dumps(feature_list, sort_keys=True))


def main(argv=None):
    query('Normal.jpg','n.txt',10)


if __name__ == '__main__':
	main()

