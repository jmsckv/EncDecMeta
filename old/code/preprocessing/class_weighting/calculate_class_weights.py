import os

import numpy as np
import tqdm
from PIL import Image

from code.model import defaults
from old_utils.utils.serialization import dict_to_yaml

DEFAULT_CLASS_WEIGHTING = defaults['loss']['class_weighting_constant']


def calculate_class_weights(in_path, out_path, dataset):
    print(f'Calculating class weights on labels in dir: {in_path}')
    num_classes = defaults['dataset'][dataset]['n_cl']
    trainId_to_count = {trainID: float(0) for trainID in range(num_classes)}

    # get the total number of pixels in all train images that are of each object class

    for label_img_path in tqdm.tqdm(os.listdir(in_path)):
        if dataset == 'Chargrid':
            label_img = np.load(os.path.join(in_path, label_img_path))['labels']  # stored as .npz
        else:
            label_img = Image.open(os.path.join(in_path, label_img_path))  # stored as .png
        for trainId in range(num_classes):
            # count how many pixels in label_img which are of object class trainId
            trainId_mask = np.equal(label_img, trainId)
            trainId_count = np.sum(trainId_mask)
            # add to the total count:
            trainId_to_count[trainId] += trainId_count

        # compute the class weights according to the ENet paper
        class_weights = {i: None for i in range(len(trainId_to_count))}
        total_count = sum(trainId_to_count.values())
        assert total_count > 0, 'Should not calculate a class weight if no pixel belonging to this class has been observed.'
        for trainId, count in trainId_to_count.items():
            trainId_prob = float(count) / float(total_count)
            trainId_weight = 1 / np.log(DEFAULT_CLASS_WEIGHTING + trainId_prob)
            class_weights[trainId] = float(trainId_weight)

    # serialize weights and further information on disk
    dict_to_yaml(class_weights, 'class_weights.yaml', out_path)
