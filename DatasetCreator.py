import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "/home/benjamin/Datasets/GA"
CATEGORIES = ['H', 'S', 'U']
#CATEGORIES = ['H']

IMG_SIZE = 64

training_data = []

def create_training_data():
    n = 0
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)          #labeling???
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
                if n%100==0:
                    print(n)
                n += 1
            except Exception as e:
                pass

        random.shuffle(training_data)

        X = []
        y = []

        for features, label in training_data:
            X.append(features)
            y.append(label)

        X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        pickle_out = open("X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

        #pickle_in = open("X.pickle", "rb")         # READ DATA
        #X = pickle.load(pickle_in)

''' #SHOW IMAGES
        plt.imshow(new_array, cmap="gray")
        plt.show()
        break
    break
    
print(img_array)
'''

create_training_data()

print("length = " + str(len(training_data)))

'''
for sample in training_data[:10]:
    print(sample[1])
'''
