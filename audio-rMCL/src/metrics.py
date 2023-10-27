import numpy as np
import torch
from utils import compute_spherical_distance
import sys
import h5py
import os
import cv2
eps = 1e-2

def frame_recall(predictions: torch.Tensor,
                 targets: torch.Tensor, mh_mode = False, conf_mode = False) -> torch.Tensor:
    """Frame-recall metric, describing the percentage of frames where the number of predicted sources matches the number
    of sources provided in the ground-truth data. For additional information, refer to e.g.

    Adavanne et al.: "A multi-room reverberant dataset for sound event localization and detection" (2019)

    Args: 
        predictions: predicted source activities and doas
        targets: ground-truth source activities and doas

    Returns:
        frame recall.
    """
    if mh_mode :
        if conf_mode : 
            hyps_DOAs_pred_stacked, conf_stacked = predictions #Shape ([batch,T,self.num_hypothesis,2],[batch,T,self.num_hypothesis,1])            
        else :
            hyps_DOAs_pred_stacked, _ = predictions #Shape [batch,T,self.num_hypothesis,2]
            conf_stacked = torch.ones_like(hyps_DOAs_pred_stacked[:,:,:,:1]) #Shape [batch,T,self.num_hypothesis,1]

        assert conf_stacked.shape == hyps_DOAs_pred_stacked.shape[:-1] + (1,)
        conf_stacked = conf_stacked.squeeze(-1).cpu()
        predicted_num_active_sources = torch.round(torch.sum(conf_stacked, dim=-1))

    else : 
        predicted_source_activity = predictions[0].cpu()
        predicted_num_active_sources = torch.sum(torch.sigmoid(predicted_source_activity) > 0.5, dim=-1)

    target_source_activity = targets[0].cpu()
    target_num_active_sources = torch.sum(target_source_activity, dim=-1)

    frame_recall = torch.mean((predicted_num_active_sources == target_num_active_sources).float())

    return frame_recall

def compute_spherical_distance_np(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes the distance between two points (given as angles) on a sphere, as described in Eq. (6) in the paper.

    Args:
        y_pred (np.ndarray): Numpy array of predicted azimuth and elevation angles.
        y_true (np.ndarray): Numpy array of ground-truth azimuth and elevation angles.

    Returns:
        np.ndarray: Numpy array of spherical distances.
    """
    if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
        raise RuntimeError('Input arrays require a dimension of two.')

    sine_term = np.sin(y_pred[:, 0]) * np.sin(y_true[:, 0])
    cosine_term = np.cos(y_pred[:, 0]) * np.cos(y_true[:, 0]) * np.cos(y_true[:, 1] - y_pred[:, 1])

    return np.arccos(np.clip(sine_term + cosine_term, a_min=-1, a_max=1))

def oracle_doa_error(predictions, targets, distance='euclidean', dataset_idx = 0, batch_idx = 0, activity_mode=False,rad2deg=True, num_sources_per_sample_min=0, num_sources_per_sample_max=3) :
    """The oracle DOA error compute the minimum distance between the hypothesis predicted and the 
    ground truth, averaged for each ground truth. 

    Args:
        predictions (torch.Tensor): Tensor of shape [batch,T,self.num_hypothesis,2]
        targets (torch.Tensor,torch.Tensor): Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
        distance (str): Distance to use. Defaults to 'euclidean'.
        dataset_idx (int): Dataset index. Defaults to 0.
        batch_idx (int): Batch index. Defaults to 0.
        activity_mode (bool): Whether the metrics computation is performed for the PIT baseline or not.
        rad2deg (bool): Whether the metric is computed in degrees or in radians. Default to degrees.
        num_sources_per_sample_min (int): Integer such that the metric is computed only for a number of sources > num_sources_per_sample_min.
        num_sources_per_sample_max (int): Integer such that the metric is computed only for a number of sources <= num_sources_per_sample_min.

    Return: 
        oracle_doa_error (torch.tensor)
    """
    if activity_mode : 
        hyps_DOAs_pred_stacked, act_pred_stacked = predictions
    else : 
        hyps_DOAs_pred_stacked, _ = predictions #Shape [batch,T,self.num_hypothesis,2]

    source_activity_target, direction_of_arrival_target = targets #Shape [batch,T,Max_sources],[batch,T,Max_sources,2]

    batch, T, num_hyps, _ = hyps_DOAs_pred_stacked.shape
    Max_sources = source_activity_target.shape[2]
    doa_error_matrix_new = np.zeros((T,1))
    count_number_actives_predictions = 0 # This variable counts the number of predictions (for element in the batch and each time step) that are active

    for t in range(T) : 
        
        hyps_stacked_t = hyps_DOAs_pred_stacked[:,t,:,:] # Shape [batch,num_hyps,2]
        source_activity_target_t = source_activity_target[:,t,:] # Shape [batch,Max_sources]
        direction_of_arrival_target_t = direction_of_arrival_target[:,t,:,:] # Shape [batch,Max_sources,2]
        
        filling_value = 10000 #Large number (on purpose) ; computational trick to ignore the "fake" ground truths.
        # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings. 
        
        #1st padding related to the inactive sources, not considered in the error calculation (with high error values)
        mask_inactive_sources = source_activity_target_t == 0
        mask_inactive_sources_target = mask_inactive_sources.unsqueeze(-1)

        if activity_mode : 
            mask_inactives_source_predicted = act_pred_stacked[:,t,:,:] == 0 # [batch,num_hyps,1], num_hyps = Max_sources in this case
            mask_inactive_sources = torch.logical_or(mask_inactive_sources_target,mask_inactives_source_predicted) # In this case, the unconsidered sources are those which are either not active in the ground truth or not active in the prediction.
        
        if mask_inactive_sources.dim()==2 : 
            mask_inactive_sources = mask_inactive_sources.unsqueeze(-1)
        mask_inactive_sources = mask_inactive_sources.expand_as(direction_of_arrival_target_t)
        direction_of_arrival_target_t[mask_inactive_sources] = filling_value #Shape [batch,Max_sources,2]

        #The ground truth tensor created is of shape [batch,Max_sources,num_hyps,2], such that each of the 
        # tensors gts[batch,i,num_hypothesis,2] contains duplicates of direction_of_arrival_target_t along the num_hypothesis
        # dimension. Note that for some values of i, gts[batch,i,num_hypothesis,2] may contain inactive sources, and therefore 
        # gts[batch,i,j,2] will be filled with filling_value (defined above) for each j in the hypothesis dimension.
        gts =  direction_of_arrival_target_t.unsqueeze(2).repeat(1,1,num_hyps,1) #Shape [batch,Max_sources,num_hypothesis,2]
        
        assert gts.shape==(batch,Max_sources,num_hyps,2)
        
        #We duplicate the hyps_stacked with a new dimension of shape Max_sources
        hyps_stacked_t_duplicated = hyps_stacked_t.unsqueeze(1).repeat(1,Max_sources,1,1) #Shape [batch,Max_sources,num_hypothesis,2]

        assert hyps_stacked_t_duplicated.shape==(batch,Max_sources,num_hyps,2)

        epsilon = 0.05
        eps = 0.001
        
        if distance=='euclidean' :
            #### With euclidean distance
            diff = torch.square(hyps_stacked_t_duplicated-gts) #Shape [batch,Max_sources,num_hypothesis,2]
            channels_sum = torch.sum(diff, dim=3) #Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]
            dist_matrix = torch.sqrt(channels_sum + eps)  #Distance matrix [batch,Max_sources,num_hypothesis]

            wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources]
            mask = wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with
            wta_dist_matrix = wta_dist_matrix*mask #[batch,Max_sources], we select only the active sources. 
            
        elif distance == 'spherical' :

            dist_matrix_euclidean = torch.sqrt(torch.sum(torch.square(hyps_stacked_t_duplicated-gts),dim=3))

            ### With spherical distance
            hyps_stacked_t_duplicated = hyps_stacked_t_duplicated.view(-1,2) #Shape [batch*num_hyps*Max_sources,2]
            gts = gts.view(-1,2) #Shape [batch*num_hyps*Max_sources,2]
            diff = compute_spherical_distance(hyps_stacked_t_duplicated,gts)
            dist_matrix = diff.view(batch,Max_sources,num_hyps) # Shape [batch,Max_sources,num_hyps]
            if activity_mode : 
                mask_inactives_source_predicted = mask_inactives_source_predicted.repeat(1,1,Max_sources)
                dist_matrix[mask_inactives_source_predicted] = filling_value #We fill the inactive sources with a large value

        wta_dist_matrix, _ = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources]
        if distance == 'spherical' : 
            eucl_wta_dist_matrix, _ = torch.min(dist_matrix_euclidean, dim=2) #wta_dist_matrix of shape [batch,Max_sources] for mask purpose
            mask = eucl_wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with
        else : 
            mask = wta_dist_matrix <= filling_value/2
        wta_dist_matrix = wta_dist_matrix*mask #[batch,Max_sources], we select only the active sources. 
        count_non_zeros = torch.sum(mask!=0) #We count the number of actives sources for the computation of the mean (below).       

        # Count the number of samples in the batch considered for this timestep (For the new version of the DOA error computation)
        num_sources_per_sample = torch.sum(source_activity_target_t.float(),dim=1,keepdim=True).repeat(1,Max_sources) # [batch,Max_sources]
        mask_where = torch.logical_and(num_sources_per_sample>num_sources_per_sample_min,num_sources_per_sample<=num_sources_per_sample_max)
        count_number_actives_predictions += torch.sum(torch.sum(mask*mask_where != 0, dim=1)!=0)

        if count_non_zeros>0 : 

            num_sources_per_sample = torch.sum(source_activity_target_t.float(),dim=1,keepdim=True).repeat(1,Max_sources) # [batch,Max_sources]
            assert num_sources_per_sample.shape == (batch,Max_sources)
            # assert torch.sum(num_sources_per_sample)/Max_sources == torch.sum(source_activity_target_t)
            wta_dist_matrix = torch.where(torch.logical_and(num_sources_per_sample>num_sources_per_sample_min,num_sources_per_sample<=num_sources_per_sample_max),wta_dist_matrix/num_sources_per_sample, torch.zeros_like(num_sources_per_sample)) #We divide by the number of active sources to get the mean error per sample
            oracle_err_new = torch.sum(wta_dist_matrix) #We compute the mean of the diff.  
            if rad2deg is True : 
                doa_error_matrix_new[t,0] = np.rad2deg(oracle_err_new.detach().cpu().numpy())
            else :
                doa_error_matrix_new[t,0] = oracle_err_new.detach().cpu().numpy()
        
        else :
            doa_error_matrix_new[t,0] = np.nan
        
    return torch.tensor(np.nansum(doa_error_matrix_new, dtype=np.float32))/count_number_actives_predictions

def create_cost_matrix(signature1, signature2, distance_func):
    num_rows = signature1.shape[0]
    num_cols = signature2.shape[0]

    cost_matrix = np.zeros((num_rows, num_cols), dtype=np.float32)

    for i in range(num_rows):
        for j in range(num_cols):
            cost_matrix[i, j] = distance_func(signature1[i, 1:].reshape(1,-1), signature2[j, 1:].reshape(1,-1))

    return cost_matrix

def compute_spherical_distance_np(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes the distance between two points (given as angles) on a sphere, as described in Eq. (6) in the paper.

    Args:
        y_pred (np.ndarray): Numpy array of predicted azimuth and elevation angles.
        y_true (np.ndarray): Numpy array of ground-truth azimuth and elevation angles.

    Returns:
        np.ndarray: Numpy array of spherical distances.
    """
    if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
        raise RuntimeError('Input arrays require a dimension of two.')

    sine_term = np.sin(y_pred[:, 0]) * np.sin(y_true[:, 0])
    cosine_term = np.cos(y_pred[:, 0]) * np.cos(y_true[:, 0]) * np.cos(y_true[:, 1] - y_pred[:, 1])

    return np.arccos(np.clip(sine_term + cosine_term, a_min=-1, a_max=1))

def sigmoid(T) :
    return 1/(1+np.exp(-T))

def emd_metric(predictions, targets, conf_mode=True, distance='euclidean', dataset_idx = 0, batch_idx = 0,rad2deg=True,num_sources_per_sample_min=0,num_sources_per_sample_max=3) :
    """Compute the EMD (or Wasserstein-2 metric) metric between the multihypothesis predictions, viewed as a mixture of diracs, and the ground truth,
    also viewed as a mixture of diracs with number of modes equal to the number of sources.

    Args:
        (in conf_mode) predictions (torch.Tensor,torch.Tensor): Shape [batch,T,self.num_hypothesis,2],[batch,T,self.num_hypothesis,1]
        targets (torch.Tensor,torch.Tensor): Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
        conf_mode (bool): If True, the predictions are in the form (hypothesis_DOAs, hypothesis_confidences), otherwise the predictions are in the form (hypothesis_DOAs). Default: True.
        distance (str): If 'euclidean', the distance between the diracs is computed using the euclidean distance. Default: 'euclidean'.
        dataset_idx (int): Index of the dataset. Default: 0.
        batch_idx (int): Index of the batch. Default: 0.
        rad2deg (bool): Whether the metric is computed in degrees or in radians. Default to degrees.
        num_sources_per_sample_min (int): Integer such that the metric is computed only for a number of sources > num_sources_per_sample_min.
        num_sources_per_sample_max (int): Integer such that the metric is computed only for a number of sources <= num_sources_per_sample_min.

    Return: 
       emd distance (torch.tensor)"""
    if conf_mode is True : 
        hyps_DOAs_pred_stacked, conf_stacked = predictions #Shape ([batch,T,self.num_hypothesis,2],[batch,T,self.num_hypothesis,1])
    else :
        hyps_DOAs_pred_stacked, _ = predictions #Shape [batch,T,self.num_hypothesis,2]
        conf_stacked = torch.ones_like(hyps_DOAs_pred_stacked[:,:,:,:1]) #Shape [batch,T,self.num_hypothesis,1]
    
    source_activity_target, direction_of_arrival_target = targets #Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
    
    #Convert tensor to numpy arrays and ensure they are float 32
    source_activity_target = source_activity_target.detach().cpu().numpy().astype(np.float32)
    conf_stacked = conf_stacked.detach().cpu().numpy().astype(np.float32) #Shape [batch,T,self.num_hypothesis,1]

    direction_of_arrival_target = direction_of_arrival_target.detach().cpu().numpy().astype(np.float32)
    hyps_DOAs_pred_stacked = hyps_DOAs_pred_stacked.detach().cpu().numpy().astype(np.float32)

    batch, T, num_hyps, _ = hyps_DOAs_pred_stacked.shape
    Max_sources = source_activity_target.shape[2]
    emd_matrix = np.zeros((T,batch))

    conf_sum = conf_stacked.sum(axis=2, keepdims=True) # [batch,T,num_hypothesis,1] (constant in the num_hypothesis axis)
    conf_stacked_normalized = np.divide(conf_stacked, conf_sum, out=np.full_like(conf_stacked, np.nan), where=conf_sum != 0)
    source_activity_target_sum = source_activity_target.sum(axis=2, keepdims=True) #Shape [batch,T,Max_sources] (constant in the Max_sources axis)
    source_activity_target_normalized = np.divide(source_activity_target, source_activity_target_sum, out=np.full_like(source_activity_target, np.nan), where=source_activity_target_sum != 0)

    signature_source = np.concatenate((conf_stacked_normalized, hyps_DOAs_pred_stacked), axis=3) # [batch,T,self.num_hypothesis,3]
    signature_target = np.concatenate((source_activity_target_normalized[:, :, :, None], direction_of_arrival_target), axis=3) # [batch,T,Max_sources,3]

    for t in range(T) :
        for number_in_batch in range(batch) :
            if source_activity_target_sum[number_in_batch, t].sum() <= num_sources_per_sample_min or source_activity_target_sum[number_in_batch, t].sum() > num_sources_per_sample_max :
                emd_matrix[t,number_in_batch] = np.nan
            elif conf_sum[number_in_batch, t].sum() == 0:
                emd_matrix[t,number_in_batch] = np.nan
            else : 
                if distance=='euclidean' :
                    #### With euclidean distance
                    emd  = cv2.EMD(signature_source[number_in_batch,t],signature_target[number_in_batch,t],cv2.DIST_L2)
                    emd_matrix[t,number_in_batch] = emd[0]
                elif distance=='spherical' :
                    #### With spherical distance
                    cost_matrix = create_cost_matrix(signature_source[number_in_batch, t], signature_target[number_in_batch, t], compute_spherical_distance_np)
                    emd  = cv2.EMD(signature_source[number_in_batch,t],signature_target[number_in_batch,t],cv2.DIST_USER, cost_matrix)
                    if rad2deg is True : 
                        emd_matrix[t,number_in_batch] = np.rad2deg(emd[0])
                    else : 
                        emd_matrix[t,number_in_batch] = emd[0]

    return torch.tensor(np.nanmean(emd_matrix, dtype=np.float32)), torch.tensor(np.nanstd(emd_matrix, dtype=np.float32))