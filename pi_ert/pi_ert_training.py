from copy import deepcopy
import matplotlib
from pi_ert.util import *
import hickle
from pi_ert.cascade_forest import CascadeForestBuilder
from menpo.io import export_image
import matplotlib.pyplot as plt
import glob
import cv2
import random
# import torch

random.seed(1234)
matplotlib.use('TkAgg')


def training(file_path, group, forests, trees):
    images = read_images(file_path, group, True)

    train_gt_shapes = get_gt_shapes(images)
    train_boxes = get_bounding_boxes(images, train_gt_shapes)

    cascade_forest_builder = CascadeForestBuilder(n_landmarks=25, n_forests=forests, n_trees=trees,
                                                  tree_depth=5, n_perturbations=30, n_test_split=20, n_pixels=224,
                                                  kappa=.3,
                                                  lr=.1)

    # training model
    model = cascade_forest_builder.build(images, train_gt_shapes, train_boxes)
    hickle.dump(model, "1_final_30_pi_ert_model.hkl")


# test model
def testing(image_path, group):
    images = read_images(image_path, group, True)
    model = hickle.load("1_final_30_pi_ert_model.hkl")
    init_shapes = []
    fin_shapes = []
    
    for i in range(len(images)):
        ibug_exam_shapes = get_gt_shapes([images[i]])
        ibug_exam_boxes = get_bounding_boxes([images[i]], ibug_exam_shapes)
        init_shape, fin_shape = model.apply(images[i], [ibug_exam_boxes[0]])
        init_shapes.append(init_shape)
        fin_shapes.append(fin_shape)
    mse, mne = calculate_mean_normalized_error(images, fin_shapes, images)
    print("Mean Squared Error: ", round(mse, 3))
    print("Mean Normalized Error: ", round(mne, 4))

    fin_shapes = fin_shapes[0]
    image = images[0]
    image_gt = deepcopy(image)
    export_image(image_gt, 'image.png', overwrite=True)
    images = []
    for im in sorted(glob.glob("image.png")):
        im = cv2.imread(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        images.append(im)
    for im in [image.landmarks['landmarks'].points, fin_shapes[0].points]:
        plt.imshow(np.array(images[0]).astype(np.uint8), cmap=plt.cm.gray)
        for p, q in im:
            x = p
            y = q
            plt.scatter(x, y, color='red')
        plt.axis('off')
        if np.array_equal(im, image.landmarks['landmarks'].points):
            plt.savefig("ground_truth_landmarks", bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig("predicted_landmarks", bbox_inches='tight', pad_inches=0)
        plt.clf()
