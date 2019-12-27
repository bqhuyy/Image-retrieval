import cv2
import sklearn
import os
from sklearn.cluster import KMeans
import time, sys
import numpy as np
import pickle

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
data_path = './compact_data/Instance-Search/oxford/images'
img_train = []
num_data = len(os.listdir(data_path))
filename = []
for idx, val in enumerate(os.listdir(data_path)):
    img = cv2.imread(os.path.join(data_path, val), cv2.IMREAD_GRAYSCALE)
    img_train.append(img)
    filename.append(val)
    printProgressBar(idx + 1, num_data, prefix = 'Load images:', suffix = 'Complete', length = 50)
    
sift = cv2.xfeatures2d.SIFT_create()

def features(img, extractor):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors
  
  
def build_descriptor_list(img_list):
    global filename
    descriptor_list = np.array([])
    num_img = len(img_list)
    for idx, img in enumerate(img_list):
        try:
            kp, des = features(img, sift)
            descriptor_list = np.append(descriptor_list, np.array(des))
            printProgressBar(idx + 1, num_img, prefix = 'Extract features:', suffix = 'Complete', length = 50)
        except:
            print(filename[idx])
    return np.reshape(descriptor_list, (-1, 128))

# start time
start_time = time.time()

descriptor_list = build_descriptor_list(img_train)

# running time
extract_feature_time = time.time() - start_time
print("--- %s seconds ---" % (extract_feature_time))

# save running time
pickle.dump(extract_feature_time, open('extract_feature_time.sav', 'wb'))

kmeans_time = []
for i in range(1,41):
    # start time
    print('K = %d <<<<<' % i)
    start_time = time.time()
    
    # k means
    kmeans = KMeans(n_clusters = i, verbose=True)
    kmeans.fit(descriptor_list)
    
    # running time
    running_time = time.time() - start_time
    print("--- %s seconds ---" % (running_time))
    
    # save running time
    kmeans_time.append(running_time)
    pickle.dump(kmeans_time, open('running_time.sav', 'wb'))
    
    # save the model to disk
    filename = 'kmeans_'+str(i)+'.sav'
    pickle.dump(kmeans, open(filename, 'wb'))
