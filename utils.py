import cv2
import numpy as np

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
        
def features(img, extractor):
    keypoints, descriptors = extractor.detectAndCompute(img, None)
    return keypoints, descriptors
  
def build_descriptor_list(extractor, img_list, filename):
    descriptor_list = np.array([])
    num_img = len(img_list)
    for idx, img in enumerate(img_list):
        try:
            kp, des = features(img, extractor)
            descriptor_list = np.append(descriptor_list, np.array(des))
            printProgressBar(idx + 1, num_img, prefix = 'Extract features:', suffix = 'Complete', length = 50)
        except:
            print('Error file: ' + filename[idx])
    return np.reshape(descriptor_list, (-1, 128))