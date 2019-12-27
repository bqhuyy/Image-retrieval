import cv2
import sklearn
import os
from sklearn.cluster import MiniBatchKMeans
import time, sys
import numpy as np
import pickle
from utils import *
        
# input data
data_path = './oxford/images'
img_train = []
num_data = len(os.listdir(data_path))
filename = []
for idx, val in enumerate(os.listdir(data_path)):
    img = cv2.imread(os.path.join(data_path, val), cv2.IMREAD_GRAYSCALE)
    img_train.append(img)
    filename.append(val)
    printProgressBar(idx + 1, num_data, prefix = 'Load images:', suffix = 'Complete', length = 50)

# initialize detector and descriptor
sift = cv2.xfeatures2d.SIFT_create()

# start time
start_time = time.time()

if os.path.isfile('./checkpoint/features.sav'):
    file = open('./checkpoint/features.sav', 'rb')
    descriptor_list = pickle.load(file)
else:
    descriptor_list = build_descriptor_list(sift, img_train, filename)
    # running time
    extract_feature_time = time.time() - start_time
    print("--- %s seconds ---" % (extract_feature_time))
    # save running time
    pickle.dump(extract_feature_time, open('./checkpoint/extract_feature_time.sav', 'wb'))
    # save features
    pickle.dump(descriptor_list, open('./checkpoint/features.sav', 'wb'))

# TRAINING
kmeans_time = []
for i in range(1, 120):
    # start time
    k = 250*i
    print('K = %d <<<<<' % k)
    start_time = time.time()
    
    # k means
    kmeans = MiniBatchKMeans(n_clusters=k, verbose=True, batch_size=k*20, init_size=3*k)
    kmeans.fit(descriptor_list)
    
    # running time
    running_time = time.time() - start_time
    print("--- %s seconds ---" % (running_time))
    
    # save running time
    kmeans_time.append(running_time)
    pickle.dump(kmeans_time, open('./checkpoint/running_time.sav', 'wb'))
    
    # save the model to disk
    filename = './mini_batch_model/kmeans_'+str(k)+'.sav'
    pickle.dump(kmeans, open(filename, 'wb'))
