import numpy as np
import torch
from scipy.spatial import distance
import argparse
import os 
import random
import glob
import yaml

def load_and_merge_configs(base_config_path, override_config_path):
    """Function for loading and merging two configuration files (in yaml).

    Args:
        base_config_path: Config file (to be overriden).
        override_config_path: Overriding config file. 

    Returns:
        merged_config: Merged configuration file.
    """
    with open(base_config_path, 'r') as base_config_file:
        base_config = yaml.safe_load(base_config_file)

    with open(override_config_path, 'r') as override_config_file:
        override_config = yaml.safe_load(override_config_file)

    # Merge the configurations
    merged_config = {**base_config, **override_config}
    for key in base_config.keys():
        if key in override_config:
            merged_config[key] = {**base_config[key], **override_config[key]}

    return merged_config

def set_seed(seed):
    """Function for setting the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(state, filename):
    """Function for saving a checkpoint into a given path (filename)."""
    torch.save(state, filename)

def keep_best_checkpoint(checkpoint_dir):
    """Function for keeping only the best checkpoint file (in term of validation loss) in a given directory."""
    # Get a list of all checkpoint files in the directory
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))

    # Sort the checkpoint files by validation loss (in descending order)
    checkpoint_files.sort(key=lambda f: float(os.path.splitext(f)[0].split('_')[-1]))

    # Keep only the best checkpoint file
    if len(checkpoint_files) > 1:
        for f in checkpoint_files[1:]:
            os.remove(f)

def parse_list(input_string):
    """Function for parsing a string of comma-separated values into a list of floats."""
    # Split the input string by commas to get individual values
    values = input_string.split(',')
    # Convert each value to float and return as a list
    return [float(value) for value in values]

def str2bool(v):
    """Function for parsing a string into a boolean value."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def ensemble_predict(models_dict, data_t, device='cpu'):
    """Function for making predictions using an ensemble of models.

    Args:
        models_dict: Dictionnary of models.
        data_t: Input data.

    Returns:
        preictions: Concatenated predictions of the ensembles (shape: (data_t.shape[0], len(models_dict), 2)).
        confs_list: Concatenated confidences of the ensembles (shape: (data_t.shape[0], len(models_dict), 2)).
    """
    data_t = data_t.to(device)
    predictions = []
    confs_list = []
    for key in models_dict:
        model = models_dict[key]
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs, confs = model(data_t.float().reshape(-1, 1))
            predictions.append(outputs)
            confs_list.append(confs)

    predictions = torch.cat(predictions, axis=1)
    confs_list = torch.cat(confs_list, axis=1)

    assert predictions.shape == (data_t.shape[0], len(models_dict), 2), "The shape of the predictions is not correct."
    
    return predictions, confs_list

def forward_single_sample(model, test_loader, device):
    """Function for performing a forward pass with model on a single sample from test_loader."""
    model.eval()
    with torch.no_grad():
      for _, data in enumerate(test_loader):
        data_t = data[0].to(device)
        data_target_position = np.array(data[1].cpu())
        data_source_activity_target = np.array(data[2].cpu())

        # Forward pass
        hyps, confs = model(data_t.float().reshape(-1,1))
        hyps = np.array(hyps.cpu())
        confs = np.array(confs.cpu())
        return hyps, confs, data_target_position, data_source_activity_target, data_t

def load_checkpoint(model, checkpoint_path, device):
    """Function for loading a checkpoint into a model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model

def plot2D_samples_mat(xs, xt, G, weights_source, axis, thr=1e-8,c=[.5, .5, 1]):
    r"""Ref: https://pythonot.github.io/_modules/ot/plot.html#plot2D_samples_mat
    """
    mx = G.max()
    scale = 1
    for i in range(xs.shape[0]):
        if weights_source[i] > 0 :
            for j in range(xt.shape[0]):
                if G[i, j] / mx > thr:
                    axis.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                            alpha=G[i, j] / mx * scale, c=c)

def find_voronoi_cell(sample, vor):
    """Function for finding the Voronoi cell of a given sample."""
    min_distance = np.inf
    closest_region = -1

    for i, point in enumerate(vor.points):
        d = distance.euclidean(sample, point)
        if d < min_distance:
            min_distance = d
            closest_region = i

    return closest_region

