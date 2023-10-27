import argparse
import os
import pickle

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display
from scipy.spatial import Voronoi, voronoi_plot_2d
from torch.utils.data import DataLoader

from datasets import MultiSourcesToyDataset, ToyDataset
from models import smcl, rmcl
from utils import (
    find_voronoi_cell, str2bool, parse_list, load_checkpoint, ensemble_predict,
    forward_single_sample
)

def emd_scores_computation(model_rmcl_star, model_smcl, model_rmcl, ensemble_models, results_saving_path, n_input_samples=50, n_gt_samples_per_frame=1000) :
  """Compute the EMD scores for the different models during inference.

  Args:
      model_rmcl_star: rMCL* model.  
      model_smcl: sMCL model.
      model_rmcl: rMCL model. 
      ensemble_models: dictionnary of single hypothesis sMCL for performing ensemble prediction. 
      results_saving_path (str): Path to save the EMD scores results.
      n_input_samples (int, optional): Number of (equally spaced) input t scalars values for performing emd computation. Defaults to 50.
      n_gt_samples_per_frame (int, optional): Number of samples from the ground-truth distribution to use at each time step for the EMD computation. Defaults to 1000.
  """

  models = {'rMCL*': model_rmcl_star, 'sMCL': model_smcl, 'rMCL': model_rmcl, 'IE': ensemble_models}

  batch_size_test = 1

  test_dataset = MultiSourcesToyDataset(n_samples=n_input_samples,grid_t=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

  emd_scores = {}

  for model_name, model in models.items():
    emd_scores[model_name] = []

  # Evaluate the model on the test dataset
  with torch.no_grad():
      for _, data in enumerate(test_loader):
          # Move the input and target tensors to the device  

          data_t = data[0]

          for model_name, model in models.items():
              
              if model_name=='IE' :
                # Forward pass
                hyps_ens, confs_ens = ensemble_predict(ensemble_models,data_t,device)
                hyps = np.array(hyps_ens.cpu())
                confs = np.array(confs_ens.cpu())
              else :
                hyps, confs = model(data_t.float().reshape(-1,1).to(device))
                hyps = np.array(hyps.cpu())
                confs = np.array(confs.cpu())
              conf_stacked_normalized = confs / np.sum(confs, axis=1, keepdims=True) # [batchxself.num_hypothesisx1]
              data_t = data_t.cpu()
              for elt_in_batch in range(hyps.shape[0]) :
                  hyps_pred_stacked = hyps[elt_in_batch,:,:] # (self.num_hypothesisx2)
                  signature_pred = np.concatenate((conf_stacked_normalized[elt_in_batch,:,:], hyps_pred_stacked), axis=1).astype(np.float32) # [self.num_hypothesisx3]
                  
                  # Generation of the samples from the target distribution
                  samples_gen, mask_activity_gen = MultiSourcesToyDataset(n_samples=n_gt_samples_per_frame).generate_dataset_distribution(t=data_t[elt_in_batch], n_samples=n_gt_samples_per_frame)
                  data_target_pos = samples_gen.reshape(-1,2)
                  data_source_activity_target_stacked = mask_activity_gen.reshape(-1,1).astype(float)
                  data_source_activity_target_normalized = data_source_activity_target_stacked / np.sum(data_source_activity_target_stacked, axis=0, keepdims=True) # [Max_sourcesx1]
                  signature_target = np.concatenate((data_source_activity_target_normalized, data_target_pos), axis=1).astype(np.float32) # [Max_sourcesx3]
                  emd  = cv2.EMD(signature_pred,signature_target,cv2.DIST_L2)
                  emd_scores[model_name].append((data_t.float(),emd[0]))

      #Save the data
      with open(results_saving_path, 'wb') as f:
          pickle.dump(emd_scores, f)

def submission_plot(model_rmcl_star, model_smcl, model_rmcl, ensemble_models, list_t_values, n_samples_gt_dist, n_samples_centroids_compt, emd_results_saving_path, num_hypothesis=20, plot_ie=False) :
  """Function to plot the results similarly to those presented in the submission paper. 

  Args:
      model_rmcl_star: rMCL* model.  
      model_smcl: sMCL model.
      model_rmcl: rMCL model. 
      ensemble_models (dict): dictionnary of single hypothesis sMCL for performing ensemble prediction. 
      list_t_values (list): List of input t values for which the results are plotted.
      n_samples_gt_dist (int): Number of samples from the ground-truth distribution to plot (green points) at each time step (in list_t_values).
      n_samples_centroids_compt (int): Number of samples from the ground-truth distribution to use for the centroids computation.
      emd_results_saving_path (str): Path of the EMD scores results.
      num_hypothesis (int, optional): Number of hypothesis used. Defaults to 20.
  """

  index = 0
  fig = plt.figure(figsize=(39,15))
  gs = gridspec.GridSpec(2, 9, width_ratios=[13, 6.5, 7.8,0., 7.8,0.,7.8,0.05, 0.8],height_ratios=[1, 1])#,hspace=0.06,wspace=0.05)

  axes = [
  plt.subplot(gs[:, 0]),     
  plt.subplot(gs[0, 2]),     
  plt.subplot(gs[1, 2]),     
  plt.subplot(gs[0, 4]),     
  plt.subplot(gs[1, 4]),    
  plt.subplot(gs[0, 6]),    
  plt.subplot(gs[1, 6])    
]

  # Quantitative plot part 
  models = {'rMCL*': model_rmcl_star, 'sMCL': model_smcl, 'rMCL': model_rmcl, 'IE': ensemble_models}

  with open(emd_results_saving_path, 'rb') as f:
    emd_scores = pickle.load(f)

  axes[0].grid()
  for model_name, model in models.items():
    if model_name == 'rMCL*' :
      axes[0].scatter([np.array(elt[0].cpu()) for elt in emd_scores[model_name]], [elt[1] for elt in emd_scores[model_name]], label="rMCL*", s=150, marker='X',color='purple')
    elif model_name == 'rMCL' :
      axes[0].scatter([np.array(elt[0].cpu()) for elt in emd_scores[model_name]], [elt[1] for elt in emd_scores[model_name]], label=model_name, s=150, marker='o',color='royalblue',edgecolor='black',linewidth=0.5)
    elif model_name == 'sMCL' :
      axes[0].scatter([np.array(elt[0].cpu()) for elt in emd_scores[model_name]], [elt[1] for elt in emd_scores[model_name]], label=model_name, s=150, marker='D',color='lightcoral')  

  if plot_ie :
    axes[0].scatter([np.array(elt[0].cpu()) for elt in emd_scores['IE']], [elt[1] for elt in emd_scores['IE']], label='IE', s=150, marker='^',color='gold', edgecolors='black')  
  else : 
    axes[0].scatter([],[],s=150, marker="^", c='gold',label='IE',edgecolors='black')
  axes[0].set_xlabel('t',fontsize=30)
  axes[0].set_ylabel('emd',fontsize=30)

  i = -1

  for t in list_t_values : 

    i+=2

    # Create a DataLoader for the test dataset
    test_dataset = MultiSourcesToyDataset(n_samples=1,t=t)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # # Evaluate the models
    hyps_rmcl, confs_rmcl, _, _, _ = forward_single_sample(model_rmcl, test_loader, device)
    hyps_smcl, _, _, _, _ = forward_single_sample(model_smcl, test_loader, device)

    # Forward pass
    hyps_ens, confs_ens = ensemble_predict(ensemble_models, torch.tensor([t]),device)
    hyps_ens = np.array(hyps_ens.cpu())
    confs_ens = np.array(confs_ens.cpu())

    confs_rmcl = confs_rmcl/np.sum(confs_rmcl, axis=1, keepdims=True)
    confs_viz = confs_rmcl
    samples = ToyDataset(n_samples=n_samples_gt_dist).generate_dataset_distribution(t=t, n_samples=n_samples_gt_dist)
    axes[i].scatter(samples[:, 0], samples[:, 1], c='lightgreen',s=5)
    axes[i].scatter(hyps_ens[index,:, 0], hyps_ens[index,:, 1], marker="^", c='gold', s=200,edgecolors='black')

    # MH Part
    axes[i].scatter([hyps_smcl[index,k, 0] for k in range(num_hypothesis)], [hyps_smcl[index,k, 1] for k in range(num_hypothesis)], c='lightcoral',s=200,marker='D')

    cmap = plt.get_cmap('Blues')
    cmap_norm = plt.Normalize(vmin=np.min(confs_rmcl[index,:,:]), vmax=np.max(confs_rmcl[index,:,:]))
    colors = [cmap(cmap_norm(confs_viz[index,k,0])) for k in range(num_hypothesis)]
    axes[i].scatter([hyps_rmcl[index,k, 0] for k in range(num_hypothesis)], [hyps_rmcl[index,k, 1] for k in range(num_hypothesis)], c=colors,s=200,edgecolors='black')

    axes[i].set_xlim(-1, 1)
    axes[i].set_ylim(-1, 1)
    axes[i].set_aspect('equal')

    cax = plt.subplot(gs[:, 8])
    # Create a dummy plot for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(confs_rmcl), vmax=np.max(confs_rmcl)))
    cbar = fig.colorbar(sm, cax=cax, aspect=20)
    cbar.ax.tick_params(labelsize='xx-large')

    points = hyps_rmcl[index,:,:]
      
    # Compute the Voronoi tessellation
    vor = Voronoi(points)

    samples_centroids = ToyDataset(n_samples=n_samples_centroids_compt).generate_dataset_distribution(t=t, n_samples=n_samples_centroids_compt)

    # Plot the Voronoi diagram
    voronoi_plot_2d(vor, ax=axes[i+1],show_vertices = False,show_points=False)

    axes[i+1].scatter(samples[:, 0], samples[:, 1], c='lightgreen',s=5,label='samples GT dist')
    axes[i+1].scatter(points[:, 0], points[:, 1], marker='o',c=colors,s=200,label='hypothesis',edgecolors='black')

    points_in_cells = np.zeros(len(vor.points), dtype=int)

    # Assign each sample to its corresponding Voronoi cell
    for sample in samples_centroids:
        cell = find_voronoi_cell(sample, vor)
        points_in_cells[cell] += 1

    for cell in range(len(vor.points)):
        if points_in_cells[cell] > 0:
            samples_in_cell = []
            for sample in samples_centroids :
              if find_voronoi_cell(sample, vor) == cell:
                  samples_in_cell.append(sample)
            samples_in_cell = np.array(samples_in_cell)
            axes[i+1].scatter(np.mean(samples_in_cell[:, 0]), np.mean(samples_in_cell[:, 1]), c='red',s=10)

    # Customize plot
    axes[i+1].set_xlim(-1,1)
    axes[i+1].set_ylim(-1,1)
    axes[i+1].set_aspect('equal')
    axes[i+1].set_xlabel(f't = {t}', labelpad=10, fontsize=30)

  for i in range(1,7):
    axes[i].set_xticks([])
    axes[i].set_yticks([])

  axes[1].annotate('Predictions', xy=(0, 0.5), xytext=(-0.06, 0.5),
          rotation=90, xycoords='axes fraction', textcoords='axes fraction',
          ha='center', va='center',fontsize=25)

  # Add rotated titles using annotate for axes[2]
  axes[2].annotate('Voronoi tessellations', xy=(0, 0.5), xytext=(-0.06, 0.5),
              rotation=90, xycoords='axes fraction', textcoords='axes fraction',
              ha='center', va='center',fontsize=25)

  plt.subplots_adjust(hspace=0.09, wspace=0.01)

  axes[0].tick_params(axis='x', labelsize=20)
  axes[0].tick_params(axis='y', labelsize=20) 
  axes[1].set_zorder(-1)
  axes[0].legend(fontsize=25, loc='upper left', ncol=1, bbox_to_anchor=(1.05, 0.6), markerscale=2,columnspacing=0.)  

  display(fig)
  plt.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Argument parser for choosing test sample')
    parser.register('type', 'list', parse_list)
    parser.add_argument('--t_values', type='list', default=[0.1,0.6,0.9], help='List of t values to be used for the visualization')
    parser.add_argument('--n_gt', type=int, default=3000, help='Number of samples for the ground truth distribution in the submission plot.')
    parser.add_argument('--checkpoint_smcl', default="checkpoints/checkpoints_smcl", help='Checkpoint smcl')
    parser.add_argument('--checkpoint_rmcl_star', default="checkpoints/checkpoints_rmcl_star", help='Checkpoint rmcl star')
    parser.add_argument('--checkpoint_rmcl', default="checkpoints/checkpoints_rmcl", help='Checkpoint rmcl')
    parser.add_argument('--num_hypothesis', type=int, default=20, help='Number of hypotheses used')
    parser.add_argument('--ensemble_prediction', type=str2bool, default=True, help='Whether to perform ensemble prediction or not')
    parser.add_argument('--checkpoints_ensembles', type=str, default="checkpoints/checkpoints_ensembles_saved", help='Path of the ensemble checkpoints')
    parser.add_argument('--show_submission_plot',type=str2bool, default=True, help='Whether to plot the submission Figures or not')
    parser.add_argument('--n_samples_centroids_compt', type=int, default=35000, help='Numer of samples for the computation of the centroids in the Voronoi cells')
    parser.add_argument('--compute_emd',type=str2bool, default=False, help='Whether to perform the EMD computation or not')
    parser.add_argument('--emd_results_saving_path', type=str, default="data_saved/emd_results.pickle", help='Path where to save the EMD results')
    parser.add_argument('--emd_compt_n_input_samples', type=int, default=50, help='Number of input (t) samples in the emd computation')
    parser.add_argument('--emd_compt_n_gt_samples_per_frame', type=int, default=1000, help='Number of ground truth (2D) samples per frame in the emd computation')
    parser.add_argument('--plot_ie', type=str2bool, default=False, help='Whether to plot the IE EMD scores or not')
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for computation (cuda or cpu)")

    args = parser.parse_args()

    if len(os.listdir(args.checkpoint_smcl))==0 or len(os.listdir(args.checkpoint_rmcl))==0 :
      raise ValueError("There are no checkpoints available for comparative evaluation")

    checkpoint_path_smcl = os.path.join(args.checkpoint_smcl,os.listdir(args.checkpoint_smcl)[0])
    checkpoint_path_rmcl_star = os.path.join(args.checkpoint_rmcl_star,os.listdir(args.checkpoint_rmcl_star)[0])
    checkpoint_path_rmcl = os.path.join(args.checkpoint_rmcl,os.listdir(args.checkpoint_rmcl)[0])

    device = torch.device(args.device)

    # Instantiate models and load checkpoints
    model_smcl = load_checkpoint(smcl(num_hypothesis=args.num_hypothesis).to(device), checkpoint_path_smcl, device)
    model_rmcl_star = load_checkpoint(rmcl(num_hypothesis=args.num_hypothesis).to(device), checkpoint_path_rmcl_star, device)
    model_rmcl = load_checkpoint(rmcl(num_hypothesis=args.num_hypothesis).to(device), checkpoint_path_rmcl, device)

    #Instantiate and load checkpoint of the ensembles
    ensemble_models = {
    f'OnehypNet_{n}': load_checkpoint(smcl(num_hypothesis=1).to(device), os.path.join(args.checkpoints_ensembles, checkpoints), device)
    for n, checkpoints in enumerate(os.listdir(args.checkpoints_ensembles)) if checkpoints.endswith(".pt")
} if args.ensemble_prediction else {}
    
    if args.compute_emd is True :    
      emd_scores_computation(model_rmcl_star=model_rmcl_star,
                            model_smcl=model_smcl,
                            model_rmcl=model_rmcl, 
                            ensemble_models=ensemble_models,
                            results_saving_path = args.emd_results_saving_path,
                            n_input_samples=args.emd_compt_n_input_samples,
                            n_gt_samples_per_frame=args.emd_compt_n_gt_samples_per_frame)
    
    if args.show_submission_plot is True :
      submission_plot(
          model_rmcl_star=model_rmcl_star,
          model_smcl=model_smcl,
          model_rmcl=model_rmcl,
          ensemble_models=ensemble_models,
          list_t_values=args.t_values,
          n_samples_gt_dist=args.n_gt,
          n_samples_centroids_compt=args.n_samples_centroids_compt,
          emd_results_saving_path=args.emd_results_saving_path,
          num_hypothesis=args.num_hypothesis,
          plot_ie=args.plot_ie
      )