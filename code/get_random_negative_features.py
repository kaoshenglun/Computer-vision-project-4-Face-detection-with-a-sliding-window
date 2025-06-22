import numpy as np
import os
import random
from cyvlfeat.hog import hog
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from skimage import color
from tqdm import tqdm

# you may implement your own data augmentation functions

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    '''
    FUNC: This funciton should return negative training examples (non-faces) from
        any images in 'non_face_scn_path'. Images should be converted to grayscale,
        because the positive training data is only available in grayscale. For best
        performance, you should sample random negative examples at multiple scales.
    ARG:
        - non_face_scn_path: a string; directory contains many images which have no
                             faces in them.
        - feature_params: a dict; with keys,
                          > template_size: int (probably 36); the number of
                            pixels spanned by each train/test template.
                          > hog_cell_size: int (default 6); the number of pixels
                            in each HoG cell. 
                          Template size should be evenly divisible by hog_cell_size.
                          Smaller HoG cell sizez tend to work better, but they 
                          make things slower because the feature dimenionality 
                          increases and more importantly the step size of the 
                          classifier decreases at test time.
    RET:
        - features_neg: (N,D) ndarray; N is the number of non-faces and D is 
                        the template dimensionality, which would be, 
                        (template_size/hog_cell_size)^2 * 31,
                        if you're using default HoG parameters.
        - neg_examples: TODO
    '''
    #########################################
    ##          you code here              ##
    #########################################
    features_neg = []
    templatesize = feature_params['template_size']
    images = [os.path.join(non_face_scn_path,f) for f in os.listdir(non_face_scn_path) if '.jpg' in f]    
    neg_examples = len(images)
    for i in range(len(images)):
        img = imread(images[i],as_grey=True)
        height = img.shape[0]
        width = img.shape[1]
        img_samples = int(np.ceil(num_samples/neg_examples))
        if min(width,height)-templatesize < img_samples:
            img_samples = min(width,height)-templatesize 
        height_index = np.random.choice(height-templatesize, img_samples, replace=False)
        width_index = np.random.choice(width-templatesize, img_samples, replace=False)

        for j in range(img_samples):
            img_cell = img[height_index[j]:height_index[j]+templatesize, width_index[j]:width_index[j]+templatesize]
            hog_feat = hog(img_cell, feature_params['hog_cell_size']).flatten()
            features_neg.append(hog_feat.tolist())
    if len(features_neg) > num_samples:
        feat_ix = np.random.choice(len(features_neg),num_samples,replace=False)
        features_neg = np.asarray(features_neg)[feat_ix]   #刪除多餘的負例子
    else:
        features_neg = np.asarray(features_neg)
    #########################################
    ##          you code here              ##
    #########################################
            
    return features_neg, neg_examples

