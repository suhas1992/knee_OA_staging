
"""
Demo script showing detections in sample X-ray images.

"""
import matplotlib
matplotlib.use('Agg')

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

from sklearn import model_selection
import pickle

import sys

CLASSES = ('__background__',
           '0','1','2','3','4')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel')}


    
    
    
        
def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'test_images', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    
    return scores, boxes

def predict_vis_detect(scores, boxes, im_name, model = 'avg'):
    """ Predict and visualize detections for test images using the desired machine learning model """
    
    # Predict the label
    if model == 'avg':
        sum_scores = np.sum(scores,axis=0)
        label = np.argmax(sum_scores[1:])
        
    
    elif model == 'SVM':
        filename = 'classification_models/SVM.sav'
    elif model == 'Random_forest':
        filename = 'classification_models/Random_forest.sav'
    elif model == 'MLP':
        filename = 'classification_models/MLP_15_10.sav'
    else:
        print "Model type not found. Available models : avg, SVM, Random_forest, MLP"
        sys.exit(0)
    
    if model != 'avg':
        
        num_boxes = 300
        num_labels = 5

        scores_1 = np.reshape(scores,(1,num_boxes*(num_labels+1)))
        loaded_clf = pickle.load(open(filename,'rb'))

        label = int(loaded_clf.predict(scores_1)[0])
    
    
    # Load the image
    im_file = os.path.join(cfg.DATA_DIR, 'test_images', im_name)
    im = cv2.imread(im_file)

    
    # Predict bounding box and visualizing it 
    CONF_THRESH = 0.05
    NMS_THRESH = 0.3
    
    imp_dets = []
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        
        max_score = 0
        max_idx = 0
        for i in inds:
            score = dets[i,-1]
            if score > max_score:
                max_score = score
                max_idx = i
        imp_dets.append((cls,dets[max_idx]))
        
    
    
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    
    max_score = 0
    
    for tup in imp_dets:
        cls,dets = tup
        #print cls, dets[-1]
        if dets[-1] > max_score:
            max_score = dets[-1]
            max_dets = dets
       
    bbox = max_dets[:4]
    center = ((bbox[0]+bbox[2])/2 , (bbox[1]+bbox[3])/2) 
    score = max_dets[-1]
    
    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linewidth=3.5)
        )
    ax.text(bbox[0], bbox[1] - 2,
            '{} {:.3f}'.format(label, score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=14, color='white')

    ax.set_title(('Predicted label : {}').format(label),fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig('data/output_test_images/' + im_name)
    
    return label
        
    
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--model',dest='model', help='Machine Learning model used to predict label[SVM]',
                        default='avg')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    im_names = []
    for file in os.listdir('data/test_images'):
        if file.endswith('.jpg'):
            im_names.append(file)
    
    im_names = sorted(im_names)
    
    input_model = args.model
        
    for im_name in im_names:
        
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        scores, boxes = demo(net, im_name)
        
        label = predict_vis_detect(scores,boxes, im_name, model = input_model)
        
        print "Predicted label = {}".format(label)
        
        
        
            
