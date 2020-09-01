# import modules
import numpy as np
from numpy import random
from skimage import io

# get images from online source
original = io.imread('https://i.imgur.com/MlTLNmo.png')
cover = io.imread('https://homepages.cae.wisc.edu/~ece533/images/airplane.png')

# declare array
noise_1 = random.randint(255, size=(512,512,3), dtype=np.uint8)
noise_2 = random.randint(255, size=(512,512,3), dtype=np.uint8)
noise_3 = random.randint(255, size=(512,512,3), dtype=np.uint8)
noise_4 = np.empty([512,512,3], dtype=np.uint8)

covered_1 = np.empty([512,512,3], dtype=np.uint8)
covered_2 = np.empty([512,512,3], dtype=np.uint8)
covered_3 = np.empty([512,512,3], dtype=np.uint8)
covered_4 = np.empty([512,512,3], dtype=np.uint8)

retrieved_1 = np.empty([512,512,3], dtype=np.uint8)
retrieved_2 = np.empty([512,512,3], dtype=np.uint8)
retrieved_3 = np.empty([512,512,3], dtype=np.uint8)
retrieved_4 = np.empty([512,512,3], dtype=np.uint8)

recovered = np.empty([512,512,3], dtype=np.uint8)

# calculate noise
noise_4 = original ^ noise_1 ^ noise_2 ^ noise_3

# embed image
covered_1 = ((cover >> 4) << 4) | (noise_1 >> 4)
covered_2 = ((cover >> 4) << 4) | (noise_2 >> 4)
covered_3 = ((cover >> 4) << 4) | (noise_3 >> 4)
covered_4 = ((cover >> 4) << 4) | (noise_4 >> 4)

# retrieve noise
retrieved_1 = covered_1 << 4
retrieved_2 = covered_2 << 4
retrieved_3 = covered_3 << 4
retrieved_4 = covered_4 << 4

# recover original image
recovered = retrieved_1 ^ retrieved_2 ^ retrieved_3 ^ retrieved_4