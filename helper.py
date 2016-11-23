# Copyright (c) Bagus Cahyono 2016

# Importing Library
import numpy as np
import glob
from PIL import Image as im

def load_data():
    # Folder where normal training data at
    trn_folder_path = 'train/normal/*.jpg'
    trp_folder_path = 'train/pco/*.jpg'
    vln_folder_path = 'validation/normal/*.jpg'
    vlp_folder_path = 'validation/pco/*.jpg'
    # Get all .jpg file path
    trn_path = glob.glob(trn_folder_path)
    trp_path = glob.glob(trp_folder_path)
    vln_path = glob.glob(vln_folder_path)
    vlp_path = glob.glob(vlp_folder_path)

    # Read all .jpg file
    # Convert it to grayscale and then put in array tr
    # Each image will have 200x300 dimension
    trn = np.array([np.array(im.open(trn_path[i]).convert('L'), 'f') for i in range(len(trn_path))])
    trp = np.array([np.array(im.open(trp_path[i]).convert('L'), 'f') for i in range(len(trp_path))])
    vln = np.array([np.array(im.open(vln_path[i]).convert('L'), 'f') for i in range(len(vln_path))])
    vlp = np.array([np.array(im.open(vlp_path[i]).convert('L'), 'f') for i in range(len(vlp_path))])

    # Combine PCO and non-PCO into one data for training and validation
    tr = np.vstack((trn, trp))
    tr = tr[:, :, :, np.newaxis]
    tr_label = np.hstack((np.zeros(len(trn), dtype=np.int), np.ones(len(trp), dtype=np.int)))
    vl = np.vstack((vln, vlp))
    vl = vl[:, :, :, np.newaxis]
    vl_label = np.hstack((np.zeros(len(vln), dtype=np.int), np.ones(len(vlp), dtype=np.int)))

    # Generate permitation to shuffle all labels and data for training only
    p = np.random.permutation(len(tr))

    return tr[p], tr_label[p], vl, vl_label

def random(size):
    nin = size[0] if len(size) == 2 else size[1] * size[2] * size[3]
    nin = 6.0 / nin

    print nin
    return np.random.normal(0, nin, size)
