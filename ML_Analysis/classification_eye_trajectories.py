import tensorflow as tf
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import pickle
import getpass
import shutil
from utils.utils_ml_analysis import load_trajectories_and_labels, create_lists_for_cv_split, extract_trajectories_and_labels, create_batched_tf_dataset,\
     create_model_and_train, find_ground_truth, compute_predictions,majority_voting_across_trajectories_and_confidence_score, classification_metrics,\
     plot_roc_curve, plot_pr_curve, save_pickle_list_to_disk, load_config_file, str2bool


def ml_analysis(ds_path: str, conv_filters: list, fc_nodes: list, drop_rate: float, learning_rate: float, batch_size: int, epochs: int, ext_cv_folds: int,
                nb_random_runs: int, output_folder: str, use_x_coords: bool, network_dims: int, screen_dims: list, interpolate_coords: bool, model_to_use: str):
    """This function performs the classification of the eye-trajectories
    Args:
        ds_path (str): path to folder containing the trajectories and the labels
        conv_filters (list): number of convolutional layers to use in the CNN
        fc_nodes (list): number of fully-connected nodes to use in the CNN
        drop_rate (float): dropout rate to use for the CNN
        learning_rate (float): learning rate used during training of the CNN
        batch_size (int): batch size to use during training of the CNN
        epochs (int): number of training epochs
        ext_cv_folds (int): number of cross-validation folds
        nb_random_runs (int): number of random runs (at each run we change the cross-validation split)
        output_folder (str): path to output directory
        use_x_coords (bool): if True, we use the x coordinates of the trajectories; otherwise we use the y coordinates of the trajectories
        network_dims (int): if 1, we use the 1D-CNN; if 2, we use the 2D-CNN
        screen_dims (list): dimensions of the screen where the task is performed
        interpolate_coords (bool): only used for the 2D-CNN; it True, we interpolate the coordinates to create a unique trajectory; if False, we use only the recorded points
        model_to_use (str): name of the model that should be used for the classification
    Returns:
        None
    """

    # if output folder does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # create it
        path_config_file = sys.argv[2]  # type: str # save filename
        shutil.copyfile(path_config_file, os.path.join(output_folder, "config_file.json"))

    x_healthy, labels_healthy, subject_mapping_healthy,\
        x_neglect, labels_neglect, subject_mapping_neglect, \
            x_neglect_heminopia, labels_neglect_heminopia, subject_mapping_neglect_heminopia = load_trajectories_and_labels(ds_path,
                                                                                                                            use_x_coords,
                                                                                                                            network_dims,
                                                                                                                            screen_dims,
                                                                                                                            interpolate_coords)

    # create lists for cv split at the subject level
    all_subjects, all_labels = create_lists_for_cv_split(subject_mapping_healthy, subject_mapping_neglect, subject_mapping_neglect_heminopia)

    acc_across_runs = []
    sens_across_runs = []
    spec_across_runs = []
    ppv_across_runs = []
    npv_across_runs = []
    f1_across_runs = []
    fpr_across_runs = []
    tpr_across_runs = []
    auc_across_runs = []
    prec_across_runs = []
    rec_across_runs = []
    aupr_across_runs = []

    # we perform several random realizations (runs) of the cross-validation such that we have different train-test splits
    for seed in range(nb_random_runs):
        print("\n\n ------------------------- Started random realization {}".format(seed+1))
        # BEGIN CROSS VALIDATION
        # since the dataset is very imbalanced, apply stratified cross validation
        ext_skf = StratifiedKFold(n_splits=ext_cv_folds, shuffle=True, random_state=seed)
        cv_fold = 0  # type: int # counter to keep track of cross-validation fold
        all_test_subs_predictions = []  # type
        all_test_subs_ground_truths = []
        dict_probabilistic_score = {}
        for ext_train_idxs, test_idxs in ext_skf.split(all_subjects, all_labels):
            cv_fold += 1
            print("Started external CV fold {}".format(cv_fold))

            ext_train_subjects = [all_subjects[i] for i in ext_train_idxs]
            test_subjects = [all_subjects[i] for i in test_idxs]

            train_trajectories, train_labels = extract_trajectories_and_labels(subject_mapping_healthy, subject_mapping_neglect, subject_mapping_neglect_heminopia, ext_train_subjects,
                                                                               x_healthy, x_neglect, x_neglect_heminopia, labels_healthy, labels_neglect, labels_neglect_heminopia, network_dims)

            # --------------------- train the model ---------------------
            batched_train_dataset = create_batched_tf_dataset(train_trajectories, train_labels, batch_size, shuffle=True)  # create tf dataset

            if network_dims == 1:
                inputs = tf.keras.Input(shape=(train_trajectories.shape[1], 1), name='eye_trajectory')
            elif network_dims == 2:
                inputs = tf.keras.Input(shape=(train_trajectories.shape[1], train_trajectories.shape[2], 1), name='eye_trajectory')
            else:
                raise ValueError("Unknown value for network_dims; only 1 and 2 allowed; got {} instead".format(network_dims))

            trained_model = create_model_and_train(model_to_use, inputs, conv_filters, fc_nodes, drop_rate, learning_rate, network_dims,
                                                   batched_train_dataset, epochs, train_trajectories, train_labels)

            # --------------------- inference on test subjects ---------------------
            for test_sub in test_subjects:  # predict on test subjects one by one

                # extract patient-wise label
                patient_wise_ground_truth = find_ground_truth(all_subjects, test_sub, all_labels)
                all_test_subs_ground_truths.append(patient_wise_ground_truth)

                # extract trajectories and labels
                test_trajectories, test_labels = extract_trajectories_and_labels(subject_mapping_healthy, subject_mapping_neglect, subject_mapping_neglect_heminopia, test_sub,
                                                                                 x_healthy, x_neglect, x_neglect_heminopia, labels_healthy, labels_neglect, labels_neglect_heminopia, network_dims)

                # create tf.data.Dataset for the trajectories of this test sub
                batched_test_dataset = create_batched_tf_dataset(test_trajectories, test_labels, batch_size, shuffle=False)

                # compute predictions for the trajectories of this test sub
                predictions = compute_predictions(model_to_use, trained_model, batched_test_dataset, test_trajectories)

                # compute majority voting of the trajectories, and ratio between the two classes
                patient_wise_prediction, confidence_score = majority_voting_across_trajectories_and_confidence_score(predictions, patient_wise_ground_truth, model_to_use)

                all_test_subs_predictions.append(patient_wise_prediction)  # append patient-wise prediction
                dict_probabilistic_score["{}".format(str(test_sub))] = confidence_score  # save ratio for this sub

        # plot patient-wise results
        conf_mat, acc, rec, spec, prec, npv, f1 = classification_metrics(np.asarray(all_test_subs_ground_truths), np.asarray(all_test_subs_predictions), binary=True)
        acc_across_runs.append(acc)
        sens_across_runs.append(rec)
        spec_across_runs.append(spec)
        ppv_across_runs.append(prec)
        npv_across_runs.append(npv)
        f1_across_runs.append(f1)

        # ------- plot ROC curve -------
        fpr, tpr, auc_value = plot_roc_curve(all_test_subs_ground_truths, all_test_subs_predictions, ext_cv_folds, embedding_label="doc2vec", plot=False)

        # append to external lists
        fpr_across_runs.append(fpr)
        tpr_across_runs.append(tpr)
        auc_across_runs.append(auc_value)

        # ------- plot PR curve -------
        rec_curve_values, prec_curve_values, aupr_value = plot_pr_curve(all_test_subs_ground_truths, all_test_subs_predictions, ext_cv_folds, embedding_label="doc2vec", plot=False)

        # append to external lists
        rec_across_runs.append(rec_curve_values)
        prec_across_runs.append(prec_curve_values)
        aupr_across_runs.append(aupr_value)

        # --------------------------- save dict of probabilistic ratios to disk
        a_file = open(os.path.join(output_folder, "probabilistic_score_run{}.pkl").format(seed + 1), "wb")
        pickle.dump(dict_probabilistic_score, a_file)
        a_file.close()

    # ----------------------------- end of all random runs
    print("\nNetwork dims used: {}".format(network_dims))
    if network_dims == 1:
        print("Coordinate used: {}".format("x" if use_x_coords else "y"))
    print("Model used: {}".format(model_to_use))
    print("Interpolation of trajectories: {}".format(interpolate_coords))
    print("Learning Rate: {}".format(learning_rate))
    print("Epochs: {}".format(epochs))

    print("\n-----------------------------------------------------------")
    print("Average test set results over {} runs:".format(nb_random_runs))
    print("Accuracy = {:.2f} ± {:.2f}".format(np.mean(acc_across_runs), np.std(acc_across_runs)))
    print("Sensitivity (recall) = {:.2f} ± {:.2f}".format(np.mean(sens_across_runs), np.std(sens_across_runs)))
    print("Specificity = {:.2f} ± {:.2f}".format(np.mean(spec_across_runs), np.std(spec_across_runs)))
    print("Precision (PPV) = {:.2f} ± {:.2f}".format(np.mean(ppv_across_runs), np.std(ppv_across_runs)))
    print("NPV = {:.2f} ± {:.2f}".format(np.mean(npv_across_runs), np.std(npv_across_runs)))
    print("f1-score = {:.2f} ± {:.2f}".format(np.mean(f1_across_runs), np.std(f1_across_runs)))
    print("AUC = {:.2f} ± {:.2f}".format(np.mean(auc_across_runs), np.std(auc_across_runs)))
    print("AUPR = {:.2f} ± {:.2f}".format(np.mean(aupr_across_runs), np.std(aupr_across_runs)))

    # save values to disk; we will use them to plot the ROC and PR curves
    date = (datetime.today().strftime('%b_%d_%Y'))  # save today's date
    save_pickle_list_to_disk(fpr_across_runs, out_dir=os.path.join(output_folder, "out_curve_values"), out_filename="fpr_values_{}.pkl".format(date))
    save_pickle_list_to_disk(tpr_across_runs, out_dir=os.path.join(output_folder, "out_curve_values"), out_filename="tpr_values_{}.pkl".format(date))
    save_pickle_list_to_disk(auc_across_runs, out_dir=os.path.join(output_folder, "out_curve_values"), out_filename="auc_values_{}.pkl".format(date))

    save_pickle_list_to_disk(rec_across_runs, out_dir=os.path.join(output_folder, "out_curve_values"), out_filename="recall_values_{}.pkl".format(date))
    save_pickle_list_to_disk(prec_across_runs, out_dir=os.path.join(output_folder, "out_curve_values"), out_filename="prec_values_{}.pkl".format(date))
    save_pickle_list_to_disk(aupr_across_runs, out_dir=os.path.join(output_folder, "out_curve_values"), out_filename="aupr_values_{}.pkl".format(date))


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file with argparser

    # extract input args from dictionary
    network_dims = config_dict['network_dims']  # type: int # set to 1 if we want to use the 1D CNN; set to 2 if we want to use the 2D CNN
    interpolate_coords = str2bool(config_dict['interpolate_coords'])  # type: bool  # if True, we interpolate the coordinates; otherwise we use the single points
    model_to_use = config_dict['model_to_use']  # type: str
    screen_dims = config_dict['screen_dims']
    conv_filters = config_dict['conv_filters']
    fc_nodes = config_dict['fc_nodes']
    drop_rate = config_dict['drop_rate']
    batch_size = config_dict['batch_size']
    learning_rate = config_dict['learning_rate']
    epochs = config_dict['epochs']
    ext_cv_folds = config_dict['ext_cv_folds']
    nb_random_runs = config_dict['nb_random_runs']
    use_x_coords = str2bool(config_dict["use_x_coords"])
    ds_path = config_dict["ds_path"]
    output_folder = config_dict["output_folder"]

    on_hpc_cluster = getpass.getuser() in ['to5743']  # type: bool # check if user is in list of authorized users
    if on_hpc_cluster:  # if we are running in the HPC cluster
        assert tf.test.is_built_with_cuda(), "TF was not built with CUDA"
        assert tf.config.experimental.list_physical_devices('GPU'), "A GPU is required to run this script"

    assert os.path.exists(ds_path), "{} does not exist".format(ds_path)
    assert os.path.exists(output_folder), "{} does not exist".format(output_folder)
    assert network_dims in (1, 2), "network_dims can only be 1 or 2; got {} instead".format(network_dims)
    date = datetime.today().strftime('%b_%d_%Y_%Hh%Mm')  # type: str # save today's date
    output_folder = os.path.join(output_folder, "outputs_eye_tracking_InputDims{}_Model{}_Interp{}_LR{}_coord{}_{}".format(network_dims,
                                                                                                                           model_to_use,
                                                                                                                           interpolate_coords,
                                                                                                                           str(learning_rate).replace(".", "_"),
                                                                                                                           "x" if use_x_coords else "y",
                                                                                                                           date))

    ml_analysis(ds_path, conv_filters, fc_nodes, drop_rate, learning_rate, batch_size, epochs, ext_cv_folds,
                 nb_random_runs, output_folder, use_x_coords, network_dims, screen_dims, interpolate_coords, model_to_use)


if __name__ == '__main__':
    main()
