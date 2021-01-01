import os
import sys

sys.path.extend(os.environ['PYTHONPATH'])

import tqdm
from PIL import Image
import numpy as np
import math

from old_utils.utils.serialization import dict_to_yaml


def calculate_normalization_statistics(path, out_path=None, w_img=2048, h_img=1024, incl_min_max_scaling=True,
                                       test_run=None, by_hand=True, n_channels=3):
    # we only calculated this for Cityscapes

    # to be coherent and double-check against other Pytorch implementations, we scale images before normalization to range [0-1]
    # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683/25
    # we divide by the max value, which corresponds to min-max scaling as the observed min=0 and max=255
    # note: because of numerical stability issues, we perform this step after calculating the mean/std
    # compared to the Pytorch d calculated on Imagenet, in Cityscapes the images are darker and vary less

    # note: current default_op is by hand and not with np, because this does not give a memory error on 16gb MBP > problem is calculating std
    # tested: np yields the same results (and is faster until it breaks)
    # e.g. for test_run=3
    # [[0.31990248522735654, 0.357518449900013, 0.32779485327757263], array([0.21269542, 0.20661206, 0.19956369])]
    # [[0.31990248522735654, 0.357518449900013, 0.32779485327757263], array([0.21269542, 0.20661206, 0.19956369])]

    # another potential, and not covered source of error if working with very large may be numerical instability issues
    # see here: https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf

    if by_hand:
        # step 1: determine channel-means
        print('\nCalculating channel-wise means.')
        sumRGB = np.zeros(3, dtype=float)  # sumR,sumG,sumB
        N = len(os.listdir(path)[:test_run]) * w_img * h_img - 1
        for img_path in tqdm.tqdm(os.listdir(path)[:test_run]):
            img = np.asarray(Image.open(os.path.join(path, img_path)))
            sumRGB += np.asarray([np.sum(img[:, :, i]) for i in range(n_channels)], dtype=float)
        channel_means = sumRGB / N

        # step 2: sum squared distances
        print('\nCalculating pixel-wise distances from channel-wise mean, summing over all pixels and images')
        ssdRGB = np.zeros(3, dtype=float)  # sum of squared distances
        for img_path in tqdm.tqdm(os.listdir(path)[:test_run]):
            img = np.asarray(Image.open(os.path.join(path, img_path)))
            img_mean_sub = img - channel_means
            # print(img_mean_sub)
            # print(img_mean_sub.shape)
            sq_dif = img_mean_sub * img_mean_sub
            ssdRGB += np.asarray([np.sum(sq_dif[:, :, i]) for i in range(n_channels)], dtype=float)
            # print(sq_dif)
            # print(sq_dif.shape)
        # print(ssdRGB)

        # step 3: dividing by N-1 and taking square root yields standard deviation
        channel_sds = np.sqrt(ssdRGB / N - 1)
        # print(channel_sds)
        if incl_min_max_scaling:
            channel_means = [float(i) for i in channel_means / 255]
            channel_sds = [float(i) for i in list(channel_sds * math.sqrt(1 / (255 ** 2)))]
        result = [channel_means, channel_sds]
        print('\nResults calculated "by hand" (Means/SDs):')
        print(result)

    else:  # np alternative
        print('Getting channel-wise means and standard deviations calculated on entire training set.')
        D = []
        for img_path in tqdm.tqdm(os.listdir(path)[:test_run]):
            img = np.asarray(Image.open(os.path.join(path, img_path)))
            D.append(img)
        D = np.asarray(D)
        channel_means = [D[:, :, :, i].mean() for i in range(n_channels)]
        channel_sds = [D[:, :, :, i].std() for i in range(n_channels)]
        if incl_min_max_scaling:
            channel_means = list(np.asarray(channel_means, dtype=float) / 255.)
            channel_sds = list(np.asarray(channel_sds, dtype=float) / 255.)
        result = channel_means, channel_sds
        print('\nResult not calculated by hand (Means/SDs):')
        print(result)

    if out_path:
        results_dict = dict()
        results_dict['channel_means'] = result[0]
        results_dict['channel_stds'] = result[1]
        print(f'Storing normalization statistics in YAML file in {out_path}.')
        dict_to_yaml(results_dict, fn='normalization_statistics.yaml', out_dir=out_path)

    return result

# Cityscapes
# if you run the script on the original images you should obtain these results with by_hand=True
# RGB-Means, RGB-STDs, not min-max scaled
# [[72.39239877354476, 82.90891755591463, 73.1583592224375], array([47.72607102, 48.48390277, 47.66526668])]
# # RGB-Means, RGB-STDs, min-max scaled
# [[0.2838917598962539, 0.3251330100231946, 0.2868955263625], [0.18716106284161413, 0.19013295203172015, 0.1869226144355443]]
# ([0.28389176, 0.32513301, 0.28689553]);  array([0.18716106, 0.19013295, 0.18692261])
# if images distorted, min-max over batch
# after PIL-preprocessing we get the same results, but note that the first and last channel are being swapped
# CV2 oder is BGR, PIL is RGB
# [[0.2868955263625, 0.3251330100231946, 0.2838917598962539], [0.1869226144355443, 0.19013295203172015, 0.18716106284161413]]
# on workstation - run 1
# channel_means:
# - 0.2868955263625
# - 0.3251330100231946
# - 0.2838917598962539
# channel_stds:
# - 0.18692261443554464
# - 0.19013295203171993
# - 0.18716106284161427
