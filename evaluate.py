#coding=utf-8
#通过输出top1 error和top10 error来判断训练和检索效�?
from __future__ import division
import tensorflow as tf
import numpy as np
import heapq
import alexnet
import math
import oncequery as oq
import input

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_input', 'pathology_image/Test_data',
                            """Data input directory when using a product level model(trained).""")

MOVING_AVERAGE_DECAY = 0.9999

#读取测试图片
def readTestPic(filepath):
    imageset = input.ImageSet(FLAGS.test_input,False)  # pre-load datalist
    # open a graph
    with tf.Graph().as_default() as g:
        # Build model(graph)
        # First build a input pipeline(Model's input subgraph).
        images, labels, ids = imageset.next_batch(FLAGS.batch_size)  # Dont need like alexnet.FLAGS.batch_size
        logits = alexnet.inference(images)
        # Use our model
        fc1 = g.get_tensor_by_name("fc1/fc1:0")  # *** use full name: variable_scope name + var/op name + output index
        fc2 = g.get_tensor_by_name("fc2/fc2:0")
        # softmax= tf.nn.softmax(logits)	#softmax = exp(logits) / reduce_sum(exp(logits), dim), dim=-1 means add along line.
        # use l2 normalization
        # *** see l2_normalize source code, you will understand why 1 not 0
        fc1_norm = tf.nn.l2_normalize(fc1, 1)
        fc2_norm = tf.nn.l2_normalize(fc2, 1)
        fc3_norm = tf.nn.l2_normalize(logits, 1)

        # Run our model
        steps = math.ceil(
            imageset.num_exps / FLAGS.batch_size)  # *** Maybe exist some duplicate image features, next dict op will clear it.

        print(steps, imageset.num_exps, FLAGS.batch_size)
        # Restore the moving average version of the learned variables for better effect.
        EMA = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = EMA.variables_to_restore()
        # for name in variables_to_restore:
        # 	print(name)
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
        
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)


            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            fc1_list = [];
            fc2_list = [];
            fc3_list = []
            id_list = []
            s = int(steps)
            i=0
            #while not coord.should_stop():
            for step in range(s):
                _ids, _fc1, _fc2, _fc3 = sess.run([ids, fc1_norm, fc2_norm, fc3_norm])  # return nd-array
                put_2darray(_fc1, fc1_list)
                put_2darray(_fc2, fc2_list)
                put_2darray(_fc3, fc3_list)
                # if step == steps-1:
                # with open('G:/tmp/duplicate.txt','w') as f:
                # f.write(str(_ids.tolist()))
                print(_ids.shape)
                for id in _ids.tolist():
                    print(id)
                    i+=1
                    print(i)
                    # print(id) # id is byte string, not a valid key of json
                    id_list.append(id.decode('utf-8'))

            print('-----------------------------------------------------')
            print(len(id_list))
            coord.request_stop()
            coord.join(threads)

    return list(zip(id_list,fc2_list))


def put_2darray(_2darray, li):
    _li= _2darray.tolist()
    for line in _li:
            li.append(line)

def calerrorrate(testfeatures,featurelib):
    totalfn=len(testfeatures)
    top1n=0
    top10n=0
    print(totalfn)
    for testfeature in testfeatures:
        qfeature=testfeature[1]
        dists = []

        for image_feature in list(featurelib.values()):
            dist = np.linalg.norm(np.array(qfeature) - np.array(image_feature))
            dists.append(dist)

        distlist = list(zip(list(featurelib.keys()), dists))
        top_k = heapq.nsmallest(10, distlist, key=lambda d: d[1])
        res = list(zip(*top_k))[0]

        resindex=[]
        for re in res:
            index=int(re.split('#')[-2])
            resindex.append(index)

        qfeatureindex=int(testfeature[0].split('#')[-2])

        if qfeatureindex==resindex[0]:
            top1n+=1
            top10n+=1
        elif qfeatureindex in resindex:
            top10n+=1
        else: continue
        
        print('top1n:%d ,top10n:%d'%(top1n,top10n))

    print('top1nerrate: {:.2%}'.format(top1n/ totalfn))
    print('top10nerrate: {:.2%}'.format(top10n/ totalfn))
    top1n_errate = top1n / totalfn
    top10n_errate = top10n / totalfn
    return top1n_errate, top10n_errate

def main(argv=None):
    testfeatures=readTestPic(FLAGS.test_input)
    featurelib=oq.getFeaturelib()
    top1n_errate,top10n_errate=calerrorrate(testfeatures,featurelib)
    print("top1_errate: %.2f, top10_errate:%.2f"%(top1n_errate,top10n_errate))

if __name__=='__main__':
    main()