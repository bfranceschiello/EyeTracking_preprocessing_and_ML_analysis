import argparse
import json
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import scipy
import tempfile
import imageio
from skimage import color
import shutil
import tensorflow as tf
from sklearn.utils import shuffle as shuffle_sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC


def str2bool(v: str) -> bool:
    """This function converts the input parameter into a boolean
    Args:
        v (*): input argument
    Returns:
        True: if the input argument is 'yes', 'true', 't', 'y', '1'
        False: if the input argument is 'no', 'false', 'f', 'n', '0'
    Raises:
        ValueError: if the input argument is none of the above
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def get_parser():
    """This function creates a parser for handling input arguments"""
    p = argparse.ArgumentParser(description='eye_trajectory_net')
    p.add_argument('--config', type=str, required=True, help='Path to json configuration file.')
    return p


def load_config_file():
    """This function loads the input config file
    Returns:
        config_dictionary (dict): it contains the input arguments
    """
    parser = get_parser()  # create parser
    args = parser.parse_args()  # convert argument strings to objects
    with open(args.config, 'r') as f:
        config_dictionary = json.load(f)

    return config_dictionary


def save_pickle_list_to_disk(list_to_save: list, out_dir: str, out_filename: str) -> None:
    """This function saves a list to disk
    Args:
        list_to_save (list): list that we want to save
        out_dir (str): path to output folder; will be created if not present
        out_filename (str): output filename
    Returns:
        None
    """
    if not os.path.exists(out_dir):  # if output folder does not exist
        os.makedirs(out_dir)  # create it
    open_file = open(os.path.join(out_dir, out_filename), "wb")
    pickle.dump(list_to_save, open_file)  # save list with pickle
    open_file.close()


def plot_roc_curve(flat_y_test, flat_y_pred_proba, cv_folds, embedding_label="doc2vec", plot=True):
    """This function computes FPR, TPR and AUC. Then, it plots the ROC curve
    Args:
        flat_y_test (list): labels
        flat_y_pred_proba (list): predictions
        cv_folds (int): number of folds in the cross-validation
        embedding_label (str): embedding algorithm that was used
        plot (bool): if True, the ROC curve will be displayed
    """
    fpr, tpr, _ = roc_curve(flat_y_test, flat_y_pred_proba, pos_label=1)
    tpr[0] = 0.0  # ensure that first element is 0
    tpr[-1] = 1.0  # ensure that last element is 1
    auc_roc = auc(fpr, tpr)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="b", label=r'{} (AUC = {:.2f})'.format(embedding_label, auc_roc), lw=2, alpha=.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.set_title("ROC curve; {}-fold CV".format(cv_folds), weight="bold", fontsize=15)
        ax.set_xlabel('FPR (1- specificity)', fontsize=12)
        ax.set_ylabel('TPR (sensitivity)', fontsize=12)
        ax.legend(loc="lower right")
    return fpr, tpr, auc_roc


def plot_pr_curve(flat_y_test, flat_y_pred_proba, cv_folds, embedding_label="doc2vec", plot=True):
    """This function computes precision, recall and AUPR. Then, it plots the PR curve
    Args:
        flat_y_test (list): labels
        flat_y_pred_proba (list): predictions
        cv_folds (int): number of folds in the cross-validation
        embedding_label (str): embedding algorithm that was used
        plot (bool): if True, the ROC curve will be displayed
    """
    precision, recall, _ = precision_recall_curve(flat_y_test, flat_y_pred_proba)
    aupr = auc(recall, precision)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color="g", label=r'{} (AUPR = {:.2f})'.format(embedding_label, aupr))
        ax.set_title("PR curve; {}-fold CV".format(cv_folds), weight="bold", fontsize=15)
        ax.set_xlabel('Recall (sensitivity)', fontsize=12)
        ax.set_ylabel('Precision (PPV)', fontsize=12)
        ax.legend(loc="lower left")
    return recall, precision, aupr


def classification_metrics(y_true, y_pred, binary=True):
    """This function computes some standard classification metrics for a binary problem
    Args:
        y_true (np.ndarray): ground truth labels
        y_pred (np.ndarray): predicted labels
        binary (bool): indicates whether the classification is binary (i.e. two classes) or not (i.e. more classes)
    Returns:
        conf_mat (np.ndarray): confusion matrix
        acc (float): accuracy
        rec (float): recall (i.e. sensitivity, or true positive rate)
        spec (float): specificity (i.e. true negative rate)
        prec (float): precision (i.e. positive predictive value)
        npv (float): negative predictive value
        f1 (float): F1-score (i.e. harmonic mean of precision and recall)
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    acc, rec, spec, prec, npv, f1 = None, None, None, None, None, None  # initialize all metrics to None
    if binary:
        assert conf_mat.shape == (2, 2), "Confusion Matrix does not correspond to a binary task"
        tn = conf_mat[0][0]
        fn = conf_mat[1][0]
        # tp = conf_mat[1][1]
        fp = conf_mat[0][1]

        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        spec = tn / (tn + fp)
        prec = precision_score(y_true, y_pred)
        npv = tn / (tn + fn)
        f1 = f1_score(y_true, y_pred)

    else:
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, average="weighted")
        prec = precision_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average='weighted')

    return conf_mat, acc, rec, spec, prec, npv, f1


def remove_zeros_ij_from_image(input_volume: np.ndarray) -> np.ndarray:
    """This function removes all the rows, columns and slices of the input volume that only contain zero values.
    Args:
        input_volume (np.ndarray): volume from which we want to remove zeros
    Returns:
        cropped_volume (np.ndarray): cropped volume (i.e. input volume with zeros removed)
    """
    def remove_zeros_one_coordinate(input_volume_: np.ndarray, range_spatial_dim: int, spatial_dim: int):
        idxs_nonzero_slices = []  # will the contain the indexes of all the slices that have nonzero values
        for idx in range(range_spatial_dim):  # loop over coordinate
            if spatial_dim == 0:
                one_slice = input_volume_[idx, :]
            elif spatial_dim == 1:
                one_slice = input_volume_[:, idx]
            else:
                raise ValueError("spatial_dim can only be 0 or 1. Got {} instead".format(spatial_dim))

            if np.count_nonzero(one_slice) > 0:  # if the slice has some nonzero values
                idxs_nonzero_slices.append(idx)  # save slice index

        if len(idxs_nonzero_slices) > 1:  # if there is more than one nonzero slice
            min_idx = min(idxs_nonzero_slices)
            max_idx = max(idxs_nonzero_slices)
        else:
            min_idx = 0
            max_idx = range_spatial_dim

        # retain only indexes with nonzero values from the two input volumes
        if spatial_dim == 0:
            cropped_volume_ = input_volume_[min_idx:max_idx, :]
        elif spatial_dim == 1:
            cropped_volume_ = input_volume_[:, min_idx:max_idx]
        else:
            raise ValueError("spatial_dim can only be 0 or 1. Got {} instead".format(spatial_dim))

        return cropped_volume_

    assert len(input_volume.shape) == 2, "The input volume must be 2D"

    # i coordinate
    cropped_volume = remove_zeros_one_coordinate(input_volume, input_volume.shape[0], spatial_dim=0)
    # j coordinate
    cropped_volume = remove_zeros_one_coordinate(cropped_volume, input_volume.shape[1], spatial_dim=1)

    return cropped_volume


def pad_image_to_specified_shape(input_img: np.ndarray, desired_x_dim: int, desired_y_dim: int) -> np.ndarray:
    """This function zero-pads input_img up to the specified shape (desired_x_dim, desired_y_dim)
    Args:
        input_img (np.ndarray): input image that we want to pad
        desired_x_dim (int): desired dimension 1
        desired_y_dim (int): desired dimension 2
    Returns:
        padded_img (np.ndarray): output padded image
    """

    assert len(input_img.shape) == 2, "This function is intended for 2D arrays"

    # extract dims of input image
    h = input_img.shape[0]
    w = input_img.shape[1]

    # extract padding width (before and after) for rows
    a = (desired_x_dim - h) // 2
    aa = desired_x_dim - a - h

    # extract padding width (before and after) for cols
    b = (desired_y_dim - w) // 2
    bb = desired_y_dim - b - w

    padded_img = np.pad(input_img, pad_width=((a, aa), (b, bb)), mode='constant', constant_values=0)

    return padded_img


def load_trajectories_and_labels(ds_path, use_x_coords, network_dims, screen_dims, interpolate_coords):
    labels_dict = scipy.io.loadmat(os.path.join(ds_path, 'Y_p.mat'))
    subject_mapping_dict = scipy.io.loadmat(os.path.join(ds_path, 'ID_Tr.mat'))

    labels = np.squeeze(labels_dict["Y_p"])  # extract labels
    subject_mapping = np.squeeze(subject_mapping_dict["ID_Tr"])  # it contains the mapping (i.e. it tells us, for every trajectory, to which subject the trajectory belongs to)

    # Healthy
    indxH = np.squeeze(np.argwhere((subject_mapping == 1) | (subject_mapping == 2) | (subject_mapping == 9) | (subject_mapping == 11) | (subject_mapping == 13) |
                                   (subject_mapping == 14) | (subject_mapping == 15) | (subject_mapping == 17) | (subject_mapping == 18)))  # extract trajectory indexes of healthy subs

    # Neglect
    indxN = np.squeeze(np.argwhere((subject_mapping == 19) | (subject_mapping == 20) | (subject_mapping == 21) | (subject_mapping == 22) |
                                   (subject_mapping == 23) | (subject_mapping == 7) | (subject_mapping == 24)))  # extract trajectory indexes of subs with neglect

    # Neglect + Heminopia
    indxNH = np.squeeze(np.argwhere((subject_mapping == 10) | (subject_mapping == 12) | (subject_mapping == 16) | (subject_mapping == 3) |
                                    (subject_mapping == 4) | (subject_mapping == 5) | (subject_mapping == 6) | (subject_mapping == 8)))  # extract trajectory indexes of subs with neglect heminopia

    # if we want to build a 1D CNN, we only extract one coordinate of the trajectories
    if network_dims == 1:
        # load from disk
        if use_x_coords:  # use x coordinates
            x_coords_dict = scipy.io.loadmat(os.path.join(ds_path, 'X_all.mat'))
            x_coords = x_coords_dict["X_all"]
        else:  # use y coordinates
            x_coords_dict = scipy.io.loadmat(os.path.join(ds_path, 'Y_all.mat'))
            x_coords = x_coords_dict["Y_all"]

        # extract coordinates and labels
        x_coords_trim = x_coords[:, 999:2000]  # choose the trajectory length

        # z-score normalize x coordinates
        x_coords_trim_normalized = scipy.stats.zscore(x_coords_trim, axis=1)

        x_healthy = x_coords_trim_normalized[indxH, :]  # extract only trajectories belonging to healthy participants
        x_neglect = x_coords_trim_normalized[indxN, :]  # extract only trajectories belonging to participants with neglect
        x_neglect_heminopia = x_coords_trim_normalized[indxNH, :]  # extract only trajectories belonging to participants with neglect + heminopia

    # if instead we want to build a 2D CNN, then we extract both coordinates from the trajectories and create an image with 1s on the trajectories, and 0s in the rest of the pixels
    elif network_dims == 2:
        # load x coordinates
        x_coords_dict = scipy.io.loadmat(os.path.join(ds_path, 'X_all.mat'))
        x_coords = x_coords_dict["X_all"]
        x_coords_trim = x_coords[:, 999:2000]  # choose the trajectory length (i.e. trim trajectory)

        # load y coordinates
        y_coords_dict = scipy.io.loadmat(os.path.join(ds_path, 'Y_all.mat'))
        y_coords = y_coords_dict["Y_all"]
        y_coords_trim = y_coords[:, 999:2000]  # choose the trajectory length (i.e. trim trajectory)

        nb_trajectories = y_coords_trim.shape[0]
        # len_of_trajectories = y_coords_trim.shape[1]

        # create list: each item contains the x and y coordinates stacked together
        xy_coords = [np.stack((x_coords_trim[idx, :], y_coords_trim[idx, :])) for idx in range(nb_trajectories)]

        # create empty images (one image per trajectory)
        empty_images = [np.zeros((screen_dims[1] + 1, screen_dims[0] + 1)) for _ in range(nb_trajectories)]
        dirpath = tempfile.mkdtemp()  # create tmp dir
        tmp_img_path = os.path.join(dirpath, "img.png")  # create tmp path for images

        images_with_coords_no_interpolation = []
        images_with_coords_interpolated = []

        for img, coords in zip(empty_images, xy_coords):
            # -------------- create non-interpolated image
            img[tuple(coords)] = 1  # assign 1 to the pixels of the trajectories (i.e. where the subject is looking during the task)
            img = remove_zeros_ij_from_image(img)  # crop only around trajectories
            images_with_coords_no_interpolation.append(img)

            if interpolate_coords:
                # -------------- create interpolated image
                fig, ax1 = plt.subplots()  # create figure
                ax1.plot(coords[0, :], coords[1, :])  # make plot (which interpolates the points by default)
                ax1.set_xlim([0, screen_dims[1] + 1])  # set xlim
                ax1.set_ylim([0, screen_dims[0] + 1])  # set ylim
                ax1.set_axis_off()  # remove axes
                fig.savefig(tmp_img_path)  # save the full figure in tmp dir
                imrgb = imageio.imread(tmp_img_path)  # load image as numpy array
                gray_image = color.rgb2gray(color.rgba2rgb(imrgb))  # convert from rgb to grayscale
                gray_image = np.where(gray_image == 1, 0, gray_image)  # set background to 0
                gray_image = remove_zeros_ij_from_image(gray_image)  # crop only around trajectories
                images_with_coords_interpolated.append(gray_image)
                plt.close('all')  # close all opened figures (needed otherwise many figures will remain open)

        shutil.rmtree(dirpath)  # remove tmp directory

        if interpolate_coords:  # if we want to interpolate the trajectories
            img_shapes_with_interpolation = [img.shape for img in images_with_coords_interpolated]
            max_img_shape_interp = tuple(np.max(img_shapes_with_interpolation, axis=0).astype(int))  # extract max image shape
            padded_imgs_interp = [pad_image_to_specified_shape(img, max_img_shape_interp[0], max_img_shape_interp[1]) for img in images_with_coords_interpolated]
            images_with_coords_np = np.asarray(padded_imgs_interp)  # convert list to numpy array
        else:  # if instead we don't want to interpolate the trajectories
            img_shapes_no_interpolation = [img.shape for img in images_with_coords_no_interpolation]
            max_img_shape_no_interp = tuple(np.max(img_shapes_no_interpolation, axis=0).astype(int))  # extract max image shape
            padded_imgs_no_interp = [pad_image_to_specified_shape(img, max_img_shape_no_interp[0], max_img_shape_no_interp[1]) for img in images_with_coords_no_interpolation]
            images_with_coords_np = np.asarray(padded_imgs_no_interp)  # convert list to numpy array

        x_healthy = images_with_coords_np[indxH, :, :]  # extract only images belonging to healthy participants
        x_neglect = images_with_coords_np[indxN, :, :]  # extract only images belonging to participants with neglect
        x_neglect_heminopia = images_with_coords_np[indxNH, :, :]  # extract only images belonging to participants with neglect + heminopia

    else:
        raise ValueError("Unknown value for network_dims; only 1 and 2 allowed; got {} instead".format(network_dims))

    labels_healthy = labels[indxH]  # extract corresponding labels (the labels are trajectory-wise)
    labels_neglect = labels[indxN]  # extract corresponding labels (the labels are trajectory-wise)
    labels_neglect_heminopia = labels[indxNH]  # extract corresponding labels (the labels are trajectory-wise)

    subject_mapping_healthy = subject_mapping[indxH]
    subject_mapping_neglect = subject_mapping[indxN]
    subject_mapping_neglect_heminopia = subject_mapping[indxNH]

    return x_healthy, labels_healthy, subject_mapping_healthy,\
                x_neglect, labels_neglect, subject_mapping_neglect,\
                    x_neglect_heminopia, labels_neglect_heminopia, subject_mapping_neglect_heminopia


def find_idxs_of_interest(subject_mapping_healthy, subject_mapping_neglect, subject_mapping_neglect_heminopia, mappings):

    if isinstance(mappings, np.uint8):
        mappings = [mappings]

    # find indexes of training trajectories
    idxs_healthy = [idx for idx, value in enumerate(list(subject_mapping_healthy)) if value in mappings]
    idxs_neglect = [idx for idx, value in enumerate(list(subject_mapping_neglect)) if value in mappings]
    idxs_neglect_heminopia = [idx for idx, value in enumerate(list(subject_mapping_neglect_heminopia)) if value in mappings]

    return idxs_healthy, idxs_neglect, idxs_neglect_heminopia


def create_batched_tf_dataset(trajectories, labels, batch_size, shuffle=True):
    trajectories_tensors = tf.convert_to_tensor(trajectories, dtype=tf.float32)  # convert from list to tf.Tensor
    labels_tensors = tf.convert_to_tensor(labels, dtype=tf.float32)  # convert from list to tf.Tensor
    tf_dataset = tf.data.Dataset.from_tensor_slices((trajectories_tensors, labels_tensors))  # create tf.data.Dataset

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=labels.shape[0], seed=123)  # shuffle otherwise healthy and neglect are not interleaved

    batched_tf_dataset = tf_dataset.batch(batch_size)  # divide dataset in batches

    return batched_tf_dataset


def create_compiled_cnn(inputs, conv_filters, fc_nodes, drop_rate, learning_rate, network_dims):
    if network_dims == 1:
        conv1 = tf.keras.layers.Conv1D(conv_filters[0], 3, activation='relu', padding='same', data_format="channels_last")(inputs)
        conv1 = tf.keras.layers.Conv1D(conv_filters[0], 3, activation='relu', padding='same')(conv1)
        bn1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(bn1)
        conv2 = tf.keras.layers.Conv1D(conv_filters[1], 3, activation='relu', padding='same', data_format="channels_last")(pool1)
        conv2 = tf.keras.layers.Conv1D(conv_filters[1], 3, activation='relu', padding='same')(conv2)
        bn2 = tf.keras.layers.BatchNormalization()(conv2)
        pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(bn2)
        conv3 = tf.keras.layers.Conv1D(conv_filters[2], 3, activation='relu', padding='same', data_format="channels_last")(pool2)
        conv3 = tf.keras.layers.Conv1D(conv_filters[2], 3, activation='relu', padding='same')(conv3)
        bn3 = tf.keras.layers.BatchNormalization()(conv3)
        pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(bn3)
    elif network_dims == 2:
        conv1 = tf.keras.layers.Conv2D(conv_filters[0], 3, activation='relu', padding='same', data_format="channels_last")(inputs)
        conv1 = tf.keras.layers.Conv2D(conv_filters[0], 3, activation='relu', padding='same')(conv1)
        bn1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=2)(bn1)
        conv2 = tf.keras.layers.Conv2D(conv_filters[1], 3, activation='relu', padding='same', data_format="channels_last")(pool1)
        conv2 = tf.keras.layers.Conv2D(conv_filters[1], 3, activation='relu', padding='same')(conv2)
        bn2 = tf.keras.layers.BatchNormalization()(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=2)(bn2)
        conv3 = tf.keras.layers.Conv2D(conv_filters[2], 3, activation='relu', padding='same', data_format="channels_last")(pool2)
        conv3 = tf.keras.layers.Conv2D(conv_filters[2], 3, activation='relu', padding='same')(conv3)
        bn3 = tf.keras.layers.BatchNormalization()(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=2)(bn3)
    else:
        raise ValueError("Unknown value for network_dims; only 1 and 2 allowed; got {} instead".format(network_dims))

    flatten = tf.keras.layers.Flatten()(pool3)
    dense1 = tf.keras.layers.Dense(units=fc_nodes[0], activation='relu')(flatten)
    drop1 = tf.keras.layers.Dropout(drop_rate)(dense1)
    dense2 = tf.keras.layers.Dense(units=fc_nodes[1], activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(drop_rate)(dense2)
    out = tf.keras.layers.Dense(units=fc_nodes[2], activation='sigmoid')(drop2)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1),
                  metrics=["accuracy"])

    # model.summary()  # print model summary
    return model


def create_lists_for_cv_split(subject_mapping_healthy, subject_mapping_neglect, subject_mapping_neglect_heminopia):
    id_healthy = list(np.unique(subject_mapping_healthy))  # save unique IDs of healthy subjects
    id_neglect = list(np.unique(subject_mapping_neglect))  # save unique IDs of neglect subjects
    id_neglect_heminopia = list(np.unique(subject_mapping_neglect_heminopia))  # save unique IDs of neglect heminopia subjects
    id_neglect_all = sorted(id_neglect + id_neglect_heminopia)  # save unique IDs of all neglect subjects
    all_subjects = id_healthy + id_neglect_all

    # create label list for stratified cross-validation
    labels_healthy_list = [1 for _ in range(len(id_healthy))]
    labels_neglect_all_list = [0 for _ in range(len(id_neglect_all))]
    all_labels = labels_healthy_list + labels_neglect_all_list

    return all_subjects, all_labels


def extract_trajectories_and_labels(subject_mapping_healthy, subject_mapping_neglect, subject_mapping_neglect_heminopia, subs_of_interest,
                                    x_healthy, x_neglect, x_neglect_heminopia, labels_healthy, labels_neglect, labels_neglect_heminopia, network_dims):

    # find indexes of interest
    idxs_healthy, idxs_neglect, idxs_neglect_heminopia = find_idxs_of_interest(subject_mapping_healthy, subject_mapping_neglect,
                                                                               subject_mapping_neglect_heminopia, subs_of_interest)

    # select trajectories of interest (either train or test) and merge them
    trajectories_healthy = x_healthy[idxs_healthy, :]
    trajectories_neglect = x_neglect[idxs_neglect, :]
    trajectories_neglect_heminopia = x_neglect_heminopia[idxs_neglect_heminopia, :]
    trajectories = np.concatenate((trajectories_healthy, trajectories_neglect, trajectories_neglect_heminopia), axis=0)

    # create train labels
    labels_healthy = labels_healthy[idxs_healthy]
    labels_neglect = labels_neglect[idxs_neglect]
    labels_neglect_heminopia = labels_neglect_heminopia[idxs_neglect_heminopia]
    labels = np.concatenate((labels_healthy, labels_neglect, labels_neglect_heminopia))

    assert trajectories.shape[0] == labels.shape[0], "Different number of trajectories/labels found"

    # convert label values of -1 into 0
    neglect_label = -1
    if neglect_label in labels:
        labels = np.where(labels == -1, 0, labels)

    if network_dims == 1:
        # -------------------------------- remove trajectories (i.e. rows) that contain nan values --------------------------------
        idxs_rows_with_nan = np.unique(np.argwhere(np.isnan(trajectories))[:, 0])
        trajectories_without_nans = np.delete(trajectories, idxs_rows_with_nan, axis=0)  # remove rows with nans
        labels_without_nans = np.delete(labels, idxs_rows_with_nan)  # remove corresponding labels
    elif network_dims == 2:
        trajectories_without_nans = trajectories
        labels_without_nans = labels
    else:
        raise ValueError("Unknown value for network_dims; only 1 and 2 allowed; got {} instead".format(network_dims))

    assert(np.array_equal(labels_without_nans, labels_without_nans.astype(bool))), "The label vector must be binary"

    return trajectories_without_nans, labels_without_nans


def majority_voting_across_trajectories_and_confidence_score(predictions, patient_wise_ground_truth, model_to_use):
    if model_to_use == "cnn":
        predictions_thresholded = np.asarray(np.where(predictions >= 0.5, 1, 0))  # threshold predictions
    else:
        predictions_thresholded = predictions  # the other model already outputs binary predictions

    unique, counts = np.unique(predictions_thresholded, return_counts=True)  # compute uniques and counts
    idx_ground_truth = np.where(unique == patient_wise_ground_truth)  # find index of ground-truth label
    ratio_between_labels = counts[idx_ground_truth] / np.sum(counts)  # compute ratio between correctly classified trajectories and all trajectories
    most_frequent_idx = np.argmax(counts)
    most_frequent_value = unique[most_frequent_idx]
    return most_frequent_value, ratio_between_labels


def find_ground_truth(all_subjects, test_sub, all_labels):
    idx_ground_truth_label = all_subjects.index(test_sub)
    ground_truth_label = all_labels[idx_ground_truth_label]
    return ground_truth_label


def create_model_and_train(model_to_use, inputs, conv_filters, fc_nodes, drop_rate, learning_rate, network_dims,
                           batched_train_dataset, epochs, train_trajectories, train_labels):
    # shuffle trajectories and labels
    train_trajectories_shuffled, train_labels_shuffled = shuffle_sklearn(train_trajectories, train_labels, random_state=123)

    if model_to_use == "cnn":
        # create CNN
        model = create_compiled_cnn(inputs, conv_filters, fc_nodes, drop_rate, learning_rate, network_dims)  # create CNN
        _ = model.fit(batched_train_dataset, epochs=epochs)  # train
    elif model_to_use == "svm":
        model = SVC()
    elif model_to_use == "random_forest":
        model = RandomForestClassifier()
    elif model_to_use == "adaboost":
        model = AdaBoostClassifier()
    else:
        raise ValueError("model_to_use can only be cnn, svm, random_forest, adaboost")

    if model_to_use != "cnn":
        model.fit(train_trajectories_shuffled, train_labels_shuffled)

    return model


def compute_predictions(model_to_use, model, batched_test_dataset, test_trajectories):
    if model_to_use == "cnn":
        predictions = model.predict(batched_test_dataset)
    elif model_to_use == "svm" or model_to_use == "random_forest" or model_to_use == "adaboost":
        predictions = model.predict(test_trajectories)
    else:
        raise ValueError("model_to_use can only be cnn, svm, random_forest, adaboost")

    return predictions