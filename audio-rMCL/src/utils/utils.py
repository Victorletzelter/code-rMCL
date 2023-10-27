import warnings
from importlib.util import find_spec
from typing import Callable, List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

from src.utils import pylogger, rich_utils
import torch
import numpy as np
from itertools import permutations
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from typing import Tuple

# Utils file with useful functions. 
# 
# The functions task_wrapper, extras, instantiate_callbacks, instantiate_loggers
# log_hyperparameters, get_metric_value, close_loggers, and save_file were copied from
# https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py.

log = pylogger.get_pylogger(__name__)

def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure. 

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished or failed
    - Logging the exception if occurs
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # execute the task
        try:

            # apply extra utilities
            extras(cfg)

            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:

            # save exception to `.log` file
            log.exception("")

            # when using hydra plugins like Optuna, you might want to disable raising exception
            # to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:

            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return metric_dict, object_dict

    return wrap

def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger

@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()

@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)
        
def compute_spherical_distance(y_pred: torch.Tensor,
                                   y_true: torch.Tensor) -> torch.Tensor:
        if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
            assert RuntimeError('Input tensors require a dimension of two.')

        sine_term = torch.sin(y_pred[:, 0]) * torch.sin(y_true[:, 0])
        cosine_term = torch.cos(y_pred[:, 0]) * torch.cos(y_true[:, 0]) * torch.cos(y_true[:, 1] - y_pred[:, 1])

        return torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))

def compute_angular_distance(x, y):
    """Computes the angle between two spherical direction-of-arrival points.

    Args:
        x: single direction-of-arrival, where the first column is the azimuth and second column is elevation
        y: single or multiple DoAs, where the first column is the azimuth and second column is elevation

    Return:
        angular distance.
    """
    if np.ndim(x) != 1:
        raise ValueError('First DoA must be a single value.')

    return np.arccos(np.sin(x[0]) * np.sin(y[0]) + np.cos(x[0]) * np.cos(y[0]) * np.cos(y[1] - x[1]))

def get_num_params(model):
    """Returns the number of trainable parameters of a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SELLoss(_Loss):
    """Sound event localization (SEL) loss function implemented in https://github.com/chrschy/adrenaline, the official code of the paper [B]. 
    This class was copied from https://github.com/chrschy/adrenaline/blob/master/utils.py. 

    [B] Schymura, C., Ochiai, T., Delcroix, M., Kinoshita, K., Nakatani, T., Araki, S., & Kolossa, D. (2021, January). Exploiting attention-based 
    sequence-to-sequence architectures for sound event localization. In 2020 28th European Signal Processing Conference (EUSIPCO) (pp. 231-235). IEEE."""

    __constants__ = ['reduction']

    def __init__(self,
                 max_num_sources: int,
                 alpha: float = 1.0,
                 size_average=None,
                 reduce=None,
                 reduction='mean') -> None:
        super(SELLoss, self).__init__(size_average, reduce, reduction)

        if (alpha< 0) or (alpha > 1):
            assert ValueError('The weighting parameter must be a number between 0 and 1.')

        self.alpha = alpha
        self.permutations = torch.from_numpy(np.array(list(permutations(range(max_num_sources)))))
        self.num_permutations = self.permutations.shape[0]

    @staticmethod
    def compute_spherical_distance(y_pred: torch.Tensor,
                                   y_true: torch.Tensor) -> torch.Tensor:
        if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
            assert RuntimeError('Input tensors require a dimension of two.')

        sine_term = torch.sin(y_pred[:, 0]) * torch.sin(y_true[:, 0])
        cosine_term = torch.cos(y_pred[:, 0]) * torch.cos(y_true[:, 0]) * torch.cos(y_true[:, 1] - y_pred[:, 1])

        return torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        source_activity_pred, direction_of_arrival_pred, _ = predictions
        source_activity_target, direction_of_arrival_target = targets 

        source_activity_bce_loss = F.binary_cross_entropy_with_logits(source_activity_pred, source_activity_target)

        source_activity_mask = source_activity_target.bool()

        spherical_distance = self.compute_spherical_distance(
            direction_of_arrival_pred[source_activity_mask], direction_of_arrival_target[source_activity_mask])
        direction_of_arrival_loss = self.alpha * torch.mean(spherical_distance)

        loss = source_activity_bce_loss + direction_of_arrival_loss

        meta_data = {
            'source_activity_loss': source_activity_bce_loss,
            'direction_of_arrival_loss': direction_of_arrival_loss
        }

        return loss, meta_data

### Custom losses from here 

class mhloss(_Loss):
    """Multi-hypothesis sound event localization loss. Used to compute WTA loss for a sequential prediction of the DOAs
    in the SSL task. Code inspired from https://github.com/lmb-freiburg/Multimodal-Future-Prediction [A]

    [A] Makansi, O., Ilg, E., Cicek, O., & Brox, T. (2019). Overcoming limitations of mixture density networks: A sampling and fitting framework for multimodal future prediction. 
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7144-7153).
    """

    __constants__ = ['reduction']

    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction='mean',
                 mode = 'wta',
                 top_n = 1,
                 distance = 'spherical',
                 epsilon=0.05,
                 single_target_loss=False) -> None:
        """Initialization of the mhloss class.

        Args:
            size_average (_type_, optional): See the _Loss parent class. Defaults to None.
            reduce (_type_, optional): See the _Loss parent class. Defaults to None.
            reduction (str, optional): See the _Loss parent class. Defaults to 'mean'.
            mode (str, optional): Type of winner-takes-all training performed ('wta', 'wta-relaxed' or 'wta-top-n'). Defaults to 'wta'.
            top_n (int, optional): Value of n in 'wta-top-n' variant. Defaults to 1.
            distance (str, optional): Type of underlying distance used in WTA computation. Defaults to 'sperical'.
            epsilon (float, optional):  Value of epsilon in 'wta-relaxed' variant. Defaults to 0.05.
            single_target_loss (bool, optional): Whether to perform single target update (useful in the 1 hypothesis mode) Defaults to False.
        """
        super(mhloss, self).__init__(size_average, reduce, reduction)

        self.mode = mode
        self.top_n = top_n
        self.distance = distance
        self.epsilon = epsilon
        self.single_target_loss = single_target_loss

    @staticmethod
    def compute_spherical_distance(y_pred: torch.Tensor,
                                   y_true: torch.Tensor) -> torch.Tensor:
        if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
            assert RuntimeError('Input tensors require a dimension of two.')

        sine_term = torch.sin(y_pred[:, 0]) * torch.sin(y_true[:, 0])
        cosine_term = torch.cos(y_pred[:, 0]) * torch.cos(y_true[:, 0]) * torch.cos(y_true[:, 1] - y_pred[:, 1])

        return torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass for the Multi-hypothesis Sound Event Localization Loss. 

        Args:
            predictions (torch.Tensor): Tensor of shape [batch,T,self.num_hypothesis,2]
            targets (torch.Tensor,torch.Tensor): #Shape [batch,T,Max_sources],[batch,T,Max_sources,2]

        Returns:
            loss (torch.tensor)
            meta_data (dict)
        """
        hyps_DOAs_pred_stacked, _ = predictions #Shape [batch,T,self.num_hypothesis,2]
        source_activity_target, direction_of_arrival_target = targets #Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
        T = source_activity_target.shape[1]

        losses = torch.tensor(0.)
        
        for t in range(T) : 
                
            source_activity_target_t = source_activity_target[:,t,:].detach()
            direction_of_arrival_target_t = direction_of_arrival_target[:,t,:,:].detach()
            hyps_stacked_t = hyps_DOAs_pred_stacked[:,t,:,:]
            
            loss_t=self.wta_loss(hyps_stacked_t=hyps_stacked_t, 
                                                             source_activity_target_t=source_activity_target_t, 
                                                             direction_of_arrival_target_t=direction_of_arrival_target_t, 
                                                             mode=self.mode,
                                                             top_n=self.top_n, 
                                                             distance=self.distance,
                                                             epsilon=self.epsilon,
                                                             single_target_loss=self.single_target_loss)
            losses = torch.add(losses,loss_t)

        losses = losses/T

        meta_data = {
            'MHLoss':losses
        }

        return losses, meta_data 
    
    def wta_loss(self, hyps_stacked_t, source_activity_target_t, direction_of_arrival_target_t, mode='wta',top_n=1, distance='euclidean', epsilon=0.05,single_target_loss=False):
        """Winner takes all loss computation and its variants.

        Args:
            hyps_stacked_t (torch.tensor): Input tensor of shape (batch,num_hyps,2)
            source_activity_target_t torch.tensor): Input tensor of shape (batch,Max_sources)
            direction_of_arrival_target_t (torch.tensor): Input tensor of shape (batch,Max_sources,2)
            mode (str, optional): Variant of the classical WTA chosen. Defaults to 'epe'.
            top_n (int, optional): top_n winner in the Evolving WTA mode. Defaults to 1.
            distance (str, optional): _description_. Defaults to 'euclidean'.

        Returns:
            loss (torch.tensor)
        """
        
        filling_value = 1000 #Large number (on purpose) ; computational trick to ignore the "fake" ground truths.
        # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings. 
        num_hyps = hyps_stacked_t.shape[1]
        batch = source_activity_target_t.shape[0]
        Max_sources = source_activity_target_t.shape[1]
        
        #1st padding related to the inactive sources, not considered in the error calculation (with high error values)
        mask_inactive_sources = source_activity_target_t == 0
        mask_inactive_sources = mask_inactive_sources.unsqueeze(-1).expand_as(direction_of_arrival_target_t)
        direction_of_arrival_target_t[mask_inactive_sources] = filling_value #Shape [batch,Max_sources,2]
        
        #The ground truth tensor created is of shape [batch,Max_sources,num_hyps,2], such that each of the 
        # tensors gts[batch,i,num_hypothesis,2] contains duplicates of direction_of_arrival_target_t along the num_hypothesis
        # dimension. Note that for some values of i, gts[batch,i,num_hypothesis,2] may contain inactive sources, and therefore 
        # gts[batch,i,j,2] will be filled with filling_value (defined above) for each j in the hypothesis dimension.
        gts =  direction_of_arrival_target_t.unsqueeze(2).repeat(1,1,num_hyps,1) #Shape [batch,Max_sources,num_hypothesis,2]
        
        # assert gts.shape==(batch,Max_sources,num_hyps,2)
        
        #We duplicate the hyps_stacked with a new dimension of shape Max_sources
        hyps_stacked_t_duplicated = hyps_stacked_t.unsqueeze(1).repeat(1,Max_sources,1,1) #Shape [batch,Max_sources,num_hypothesis,2]

        assert hyps_stacked_t_duplicated.shape==(batch,Max_sources,num_hyps,2)
        
        if distance=='euclidean' :
            #### With euclidean distance
            diff = torch.square(hyps_stacked_t_duplicated-gts) #Shape [batch,Max_sources,num_hypothesis,2]
            channels_sum = torch.sum(diff, dim=3) #Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]
            
            eps = 0.001
            dist_matrix = torch.sqrt(channels_sum + eps)  #Distance matrix [batch,Max_sources,num_hypothesis]
            
            # assert dist_matrix.shape == [batch,Max_sources,num_hyps]
            
        elif distance == 'spherical' :

            dist_matrix_euclidean = torch.sqrt(torch.sum(torch.square(hyps_stacked_t_duplicated-gts),dim=3)) #We also compute the euclidean distance matrix, to use it as a mask for the spherical distance computation. 

            ### With spherical distance
            hyps_stacked_t_duplicated = hyps_stacked_t_duplicated.view(-1,2) #Shape [batch*num_hyps*Max_sources,2]
            gts = gts.view(-1,2) #Shape [batch*num_hyps*Max_sources,2]
            diff = compute_spherical_distance(hyps_stacked_t_duplicated,gts)
            dist_matrix = diff.view(batch,Max_sources,num_hyps) # Shape [batch,Max_sources,num_hyps]
            dist_matrix[dist_matrix_euclidean >= filling_value/2] = filling_value #We fill the parts corresponding to false gts. 

        sum_losses = torch.tensor(0.)

        if mode == 'wta': 

            if single_target_loss==True:
                wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources] 
                wta_dist_matrix, _ = torch.min(wta_dist_matrix,dim=1) 
                wta_dist_matrix = wta_dist_matrix.unsqueeze(-1) #[batch,1]

                # assert wta_dist_matrix.shape == (batch,1)
                if distance == 'spherical' : 
                    eucl_wta_dist_matrix, _ = torch.min(dist_matrix_euclidean, dim=2) #wta_dist_matrix of shape [batch,Max_sources] for mask purpose  
                    eucl_wta_dist_matrix, _ = torch.min(eucl_wta_dist_matrix,dim=1)
                    eucl_wta_dist_matrix = eucl_wta_dist_matrix.unsqueeze(-1) #Shape [batch,1]
                    mask = eucl_wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values.
                else : 
                    mask = wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values. 
                wta_dist_matrix = wta_dist_matrix*mask #[batch,1], we select only the active sources.
                
                count_non_zeros = torch.sum(mask!=0) #We count the number of actives sources for the computation of the mean (below).

            else : 
                wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources] 
                if distance=='spherical' : 
                    eucl_wta_dist_matrix, _ = torch.min(dist_matrix_euclidean, dim=2)
                    mask = eucl_wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values.
                else : 
                    mask = wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values. 
                wta_dist_matrix = wta_dist_matrix*mask #[batch,Max_sources], we select only the active sources. 
                count_non_zeros = torch.sum(mask!=0) #We count the number of actives sources for the computation of the mean (below). 

            if count_non_zeros>0 : 
                loss = torch.sum(wta_dist_matrix)/count_non_zeros #We compute the mean of the diff. 
            else :
                loss = torch.tensor(0.)    
            
            sum_losses = torch.add(sum_losses, loss)  
            
        elif mode == 'wta-relaxed':
        
            #We compute first the loss for the "best" hypothesis. 
            
            wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources], idx_selected of shape [batch,Max_sources].
            
            # assert wta_dist_matrix.shape == (batch,Max_sources)
            # assert idx_selected.shape == (batch,Max_sources)
            
            if distance=='spherical' :
                eucl_wta_dist_matrix, _ = torch.min(dist_matrix_euclidean, dim=2)
                mask = eucl_wta_dist_matrix <= filling_value/2
            else :
                mask = wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with
            wta_dist_matrix = wta_dist_matrix*mask #Shape [batch,Max_sources] ; we select only the active sources. 
            count_non_zeros_1 = torch.sum(mask!=0) #We count the number of actives sources as a sum over the batch for the computation of the mean (below).

            if count_non_zeros_1>0 : 
                loss0 = torch.multiply(torch.sum(wta_dist_matrix)/count_non_zeros_1, 1 - epsilon) #Scalar (average with coefficient)
            else :
                loss0 = torch.tensor(0.)    

            #We then the find the other hypothesis, and compute the epsilon weighted loss for them
            
            if distance=='spherical' :
                large_mask = dist_matrix_euclidean <= filling_value
            else : 
                # At first, we remove hypothesis corresponding to "fake" ground-truth.         
                large_mask = dist_matrix <= filling_value # We remove entries corresponding to "fake"/filled ground truth in the tensor dist_matrix on
            # which the min operator was not already applied. Shape [batch,Max_sources,num_hypothesis]
            dist_matrix = dist_matrix*large_mask # Shape [batch,Max_sources,num_hypothesis].
            
            # We then remove the hypothesis selected above (with minimum dist)
            mask_selected = torch.zeros_like(dist_matrix,dtype=bool) #Shape [batch,Max_sources,num_hypothesis]
            mask_selected.scatter_(2, idx_selected.unsqueeze(-1), 1) # idx_selected new shape: [batch,Max_sources,1]. 
            # The assignement mask_selected[i,j,idx_selected[i,j]]=1 is performed. 
            # Shape of mask_selected: [batch,Max_sources,num_hypothesis]
            
            # assert mask_selected.shape == (batch,Max_sources,num_hyps)
            
            mask_selected = ~mask_selected #Shape [batch,Max_sources,num_hypothesis], we keep only the hypothesis which are not the minimum.
            dist_matrix = dist_matrix * mask_selected #Shape [batch,Max_sources,num_hypothesis]
            
            # Finally, we compute the loss
            count_non_zeros_2 = torch.sum(dist_matrix!=0)

            if count_non_zeros_2 > 0 :
                loss = torch.multiply(torch.sum(dist_matrix)/count_non_zeros_2, epsilon) #Scalar for each hyp
            else : 
                loss = torch.tensor(0.)
            
            sum_losses = torch.add(sum_losses, loss)
            sum_losses = torch.add(sum_losses, loss0)
            
        elif mode == 'wta-top-n' and top_n > 1:
            
            dist_matrix = torch.multiply(dist_matrix, -1) # Shape (batch,Max_sources,num_hyps) 
            top_k, indices = torch.topk(input=dist_matrix, k=top_n, dim=-1) #top_k of shape (batch,Max_sources,top_n), indices of shape (batch,Max_sources,top_n) 
            dist_matrix_min = torch.multiply(top_k,-1) 
            
            if distance=='spherical':
                dist_matrix_euclidean = torch.multiply(dist_matrix_euclidean, -1) # Shape (batch,Max_sources,num_hyps) 
                top_k, _ = torch.topk(input=dist_matrix_euclidean, k=top_n, dim=-1) #top_k of shape (batch,Max_sources,top_n), indices of shape (batch,Max_sources,top_n) 
                dist_matrix_min_euclidean = torch.multiply(top_k,-1) 
                mask = dist_matrix_min_euclidean <= filling_value/2
            else : 
                mask = dist_matrix_min <= filling_value/2 # We create a mask of shape [batch,Max_sources,top_n] for only selecting the actives sources, i.e. those which were not filled with fake values. 
            # assert mask[:,:,0].all() == mask[:,:,-1].all() # This mask related should be constant in the third dimension.
            
            dist_matrix_min = dist_matrix_min*mask # [batch,Max_sources,top_n], we select only the active sources.  
            # assert dist_matrix_min.shape == (batch,Max_sources,top_n)
            
            count_non_zeros = torch.sum(mask[:,:,0]!=0) # We count the number of entries (in the first two dimensions) for which the mask is different from zero. 
            
            for i in range(top_n):
                
                # assert count_non_zeros == torch.sum(mask[:,:,i]!=0) # We count the number of entries for which the mask is different from zero. 
                
                if count_non_zeros > 0 :
                    loss  = torch.multiply(torch.sum(dist_matrix_min[:, :, i])/count_non_zeros, 1.0)
                else :
                    loss = torch.tensor(0.)
                
                sum_losses = torch.add(sum_losses, loss)
                
            sum_losses = sum_losses / top_n
            
        return sum_losses
    
class rmcl_loss(_Loss):
    """rmcl_loss for Multi-hypothesis with confidence sound event localization loss. Used to perform rMCL loss for a sequential prediction of the DOAs
    in the SSL task. Code inspired from https://github.com/lmb-freiburg/Multimodal-Future-Prediction [A]

    [A] Makansi, O., Ilg, E., Cicek, O., & Brox, T. (2019). Overcoming limitations of mixture density networks: A sampling and fitting framework for multimodal future prediction. 
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7144-7153)."""

    __constants__ = ['reduction']

    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction='mean',
                 mode = 'wta',
                 top_n = 1,
                 distance = 'euclidean',
                 epsilon=0.05,
                 conf_weight = 1,
                 rejection_method = 'all',
                 number_unconfident = 3) -> None:
        """Initialization of the rmcl_loss class.

        Args:
            size_average (_type_, optional): See the _Loss parent class. Defaults to None.
            reduce (_type_, optional): See the _Loss parent class. Defaults to None.
            reduction (str, optional): See the _Loss parent class. Defaults to 'mean'.
            mode (str, optional): Type of winner-takes-all training performed ('wta', 'wta-relaxed' or 'wta-top-n'). Defaults to 'wta'.
            top_n (int, optional): Value of n in 'wta-top-n' variant. Defaults to 1.
            distance (str, optional): Type of underlying distance used in WTA computation. Defaults to 'sperical'.
            epsilon (float, optional):  Value of epsilon in 'wta-relaxed' variant. Defaults to 0.05.
            conf_weight (int, optional): Value of the confidence loss weight (beta). Defaults to 1.
            rejection_method (str, optional): Method used for negative hypothesis selection in loss computation (rMCL* version). Defaults to 'all'.
            number_unconfident (int, optional): Number of negative hypothesis used in the loss computation if rejection_method='uniform_negative', 
            should be lower that the number hypothesis.  Defaults to 3.
        """
        super(rmcl_loss, self).__init__(size_average, reduce, reduction)

        self.mode = mode
        self.top_n = top_n
        self.distance = distance
        self.epsilon = epsilon
        self.conf_weight = conf_weight
        self.rejection_method = rejection_method
        self.number_unconfident = number_unconfident

    @staticmethod
    def compute_spherical_distance(y_pred: torch.Tensor,
                                   y_true: torch.Tensor) -> torch.Tensor:
        if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
            assert RuntimeError('Input tensors require a dimension of two.')

        sine_term = torch.sin(y_pred[:, 0]) * torch.sin(y_true[:, 0])
        cosine_term = torch.cos(y_pred[:, 0]) * torch.cos(y_true[:, 0]) * torch.cos(y_true[:, 1] - y_pred[:, 1])

        return torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass for the Multi-hypothesis Sound Event Localization Loss. 

        Args:
            predictions (torch.Tensor): Tensor of shape [batch,T,self.num_hypothesis,2]
            targets (torch.Tensor,torch.Tensor): #Shape [batch,T,Max_sources],[batch,T,Max_sources,2]

        Returns:
            loss (torch.tensor)
            meta_data (dict)
        """
        hyps_DOAs_pred_stacked, conf_pred_stacked, _ = predictions #Shape ([batch,T,self.num_hypothesis,2],[batch,T,self.num_hypothesis,1])
        source_activity_target, direction_of_arrival_target = targets #Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
        T = source_activity_target.shape[1]

        losses = torch.tensor(0.)
        
        for t in range(T) : 
                
            source_activity_target_t = source_activity_target[:,t,:].detach()
            direction_of_arrival_target_t = direction_of_arrival_target[:,t,:,:].detach()
            hyps_stacked_t = hyps_DOAs_pred_stacked[:,t,:,:]
            conf_pred_stacked_t = conf_pred_stacked[:,t,:,:]
            
            loss_t=self.rmcl_loss(hyps_stacked_t=hyps_stacked_t, 
                                                        conf_pred_stacked_t=conf_pred_stacked_t,
                                                        source_activity_target_t=source_activity_target_t, 
                                                        direction_of_arrival_target_t=direction_of_arrival_target_t, 
                                                        mode=self.mode,
                                                        top_n=self.top_n, 
                                                        distance=self.distance,
                                                        epsilon=self.epsilon,
                                                        conf_weight = self.conf_weight,
                                                        rejection_method=self.rejection_method,
                                                        number_unconfident=self.number_unconfident)
            losses = torch.add(losses,loss_t)

        losses = losses/T

        meta_data = {
            'MHLoss':losses
        }

        return losses, meta_data 
    
    def rmcl_loss(self, hyps_stacked_t, conf_pred_stacked_t, source_activity_target_t, direction_of_arrival_target_t, mode='wta',top_n=1, distance='euclidean', epsilon=0.05, conf_weight = 1.,rejection_method='uniform_negative',number_unconfident = 3):
        """Winner takes all loss computation and its variants.

        Args:
            hyps_stacked_t (torch.tensor): Input tensor of shape (batch,num_hyps,2)
            source_activity_target_t torch.tensor): Input tensor of shape (batch,Max_sources)
            conf_pred_stacked_t (torch.tensor): Input tensor of shape (batch,num_hyps,1)
            direction_of_arrival_target_t (torch.tensor): Input tensor of shape (batch,Max_sources,2)
            mode (str, optional): Variant of the classical WTA chosen. Defaults to 'epe'.
            top_n (int, optional): top_n winner in the Evolving WTA mode. Defaults to 1.
            distance (str, optional): _description_. Defaults to 'euclidean'.

        Returns:
            loss (torch.tensor)
        """
        filling_value = 1000 #Large number (on purpose) ; computational trick to ignore the "fake" ground truths.
        # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings. 
        num_hyps = hyps_stacked_t.shape[1]
        batch = source_activity_target_t.shape[0]
        Max_sources = source_activity_target_t.shape[1]

        # assert num_hyps >= number_unconfident, "The number of hypothesis is too small comparing to the number of unconfident hypothesis selected in the scoring" # We check that the number of hypothesis is higher than the number of "negative" hypothesis sampled. 
        
        #1st padding related to the inactive sources, not considered in the error calculation (with high error values)
        mask_inactive_sources = source_activity_target_t == 0
        mask_inactive_sources = mask_inactive_sources.unsqueeze(-1).expand_as(direction_of_arrival_target_t)
        direction_of_arrival_target_t[mask_inactive_sources] = filling_value #Shape [batch,Max_sources,2]
        
        #The ground truth tensor created is of shape [batch,Max_sources,num_hyps,2], such that each of the 
        # tensors gts[batch,i,num_hypothesis,2] contains duplicates of direction_of_arrival_target_t along the num_hypothesis
        # dimension. Note that for some values of i, gts[batch,i,num_hypothesis,2] may contain inactive sources, and therefore 
        # gts[batch,i,j,2] will be filled with filling_value (defined above) for each j in the hypothesis dimension.
        gts =  direction_of_arrival_target_t.unsqueeze(2).repeat(1,1,num_hyps,1) #Shape [batch,Max_sources,num_hypothesis,2]
        
        # assert gts.shape==(batch,Max_sources,num_hyps,2)
        
        #We duplicate the hyps_stacked with a new dimension of shape Max_sources
        hyps_stacked_t_duplicated = hyps_stacked_t.unsqueeze(1).repeat(1,Max_sources,1,1) #Shape [batch,Max_sources,num_hypothesis,2]

        # assert hyps_stacked_t_duplicated.shape==(batch,Max_sources,num_hyps,2)

        ### Management of the confidence part
        conf_pred_stacked_t = torch.squeeze(conf_pred_stacked_t,dim=-1) #(batch,num_hyps), predicted confidence scores for each hypothesis.
        gt_conf_stacked_t = torch.zeros_like(conf_pred_stacked_t) #(batch,num_hyps), will contain the ground-truth of the confidence scores. 
        
        # assert gt_conf_stacked_t.shape == (batch,num_hyps)

        if distance=='euclidean' :
            #### With euclidean distance
            diff = torch.square(hyps_stacked_t_duplicated-gts) #Shape [batch,Max_sources,num_hyps,2]
            channels_sum = torch.sum(diff, dim=3) #Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]
            
            eps = 0.001
            dist_matrix = torch.sqrt(channels_sum + eps)  #Distance matrix [batch,Max_sources,num_hyps]
            
            # assert dist_matrix.shape == (batch,Max_sources,num_hyps)
            
        elif distance == 'spherical' :

            dist_matrix_euclidean = torch.sqrt(torch.sum(torch.square(hyps_stacked_t_duplicated-gts),dim=3)) #We also compute the euclidean distance matrix, to use it as a mask for the spherical distance computation. 

            ### With spherical distance
            hyps_stacked_t_duplicated = hyps_stacked_t_duplicated.view(-1,2) #Shape [batch,num_hyps,Max_sources,2]
            gts = gts.view(-1,2) #Shape [batch,num_hyps,Max_sources,2]
            diff = compute_spherical_distance(hyps_stacked_t_duplicated,gts)
            diff = diff.view(batch,Max_sources,num_hyps) # Shape [batch,Max_sources,num_hyps]
            dist_matrix = diff # Shape [batch,Max_sources,num_hyps]
            dist_matrix[dist_matrix_euclidean >= filling_value/2] = filling_value #We fill the parts corresponding to false gts. 
            
        sum_losses = torch.tensor(0.)

        if mode == 'wta': 
            
            # We select the best hypothesis for each source
            wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources]
            if distance=='spherical' : 
                eucl_wta_dist_matrix, _ = torch.min(dist_matrix_euclidean, dim=2)
                mask = eucl_wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values.
            else : 
                mask = wta_dist_matrix <= filling_value/2 #We create a mask of shape [batch,Max_sources] for only selecting the actives sources, i.e. those which were not filled with fake values. 

            wta_dist_matrix = wta_dist_matrix*mask #[batch,Max_sources], we select only the active sources.

            # Create tensors to index batch and Max_sources dimensions
            batch_indices = torch.arange(batch,device='cuda:0')[:, None].expand(-1, Max_sources) # Shape [batch, Max_sources]

            # We set the confidences of the selected hypotheses.
            gt_conf_stacked_t[batch_indices[mask], idx_selected[mask]] = 1 #Shape [batch,num_hyps]

            count_non_zeros = torch.sum(mask!=0) #We count the number of actives sources for the computation of the mean (below). 
            
            if count_non_zeros>0 : #If at least one source is active.
                loss = torch.sum(wta_dist_matrix)/count_non_zeros #We compute the mean of the diff.
                
                selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores which
                # will be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.   
                unselected_mask = ~selected_confidence_mask # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.  

                if rejection_method=='uniform_negative' : 
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack([torch.multinomial(unselected_mask[b_idx].float(), number_unconfident, replacement=False) for b_idx in range(batch)]) #(batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(-1, number_unconfident) #(batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    # gt_conf_stacked_t[batch_indices, unconfident_indices] = 0 #(Useless) Line added for the sake of completness. 

                elif rejection_method=='all' :

                    selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(conf_pred_stacked_t[selected_confidence_mask], gt_conf_stacked_t[selected_confidence_mask])

            else :
                loss = torch.tensor(0.) 
                confidence_loss = torch.tensor(0.)   
            
            sum_losses = torch.add(sum_losses, loss)  
            sum_losses = torch.add(sum_losses, conf_weight*confidence_loss)
            
        elif mode == 'wta-relaxed':
        
            #We compute first the loss for the "best" hypothesis but also for the others with weight epsilon.  
            
            wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources], idx_selected of shape [batch,Max_sources].
            
            if distance=='spherical' :
                eucl_wta_dist_matrix, _ = torch.min(dist_matrix_euclidean, dim=2)
                mask = eucl_wta_dist_matrix <= filling_value/2
            else : 
                mask = wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with
            wta_dist_matrix = wta_dist_matrix*mask #Shape [batch,Max_sources] ; we select only the active sources. 
            count_non_zeros_1 = torch.sum(mask!=0) #We count the number of actives sources as a sum over the batch for the computation of the mean (below).

            ### Confidence management
            # Create tensors to index batch and Max_sources dimensions
            batch_indices = torch.arange(batch)[:, None].expand(-1, Max_sources) # Shape [batch, Max_sources]

            # We set the confidence of the selected hypothesis
            gt_conf_stacked_t[batch_indices[mask], idx_selected[mask]] = 1 #Shape [batch,num_hyps]
            ###

            if count_non_zeros_1>0 : 
                loss0 = torch.multiply(torch.sum(wta_dist_matrix)/count_non_zeros_1, 1 - epsilon) #Scalar (average with coefficient)
                
                selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores which
                # will be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.   
                unselected_mask = ~selected_confidence_mask # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.  

                if rejection_method=='uniform_negative' : 
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack([torch.multinomial(unselected_mask[b_idx].float(), number_unconfident, replacement=False) for b_idx in range(batch)]) #(batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(-1, number_unconfident) #(batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    gt_conf_stacked_t[batch_indices, unconfident_indices] = 0 #(Useless) Line added for the sake of completness. 

                elif rejection_method=='all' :

                    selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(conf_pred_stacked_t[selected_confidence_mask], gt_conf_stacked_t[selected_confidence_mask])
               
            else :
                loss0 = torch.tensor(0.) 
                confidence_loss = torch.tensor(0.)

            #We then the find the other hypothesis, and compute the epsilon weighted loss for them
            
            # At first, we remove hypothesis corresponding to "fake" ground-truth.  
            if distance=='spherical' :
                large_mask = dist_matrix_euclidean <= filling_value
            else :        
                large_mask = dist_matrix <= filling_value # We remove entries corresponding to "fake"/filled ground truth in the tensor dist_matrix on
            # which the min operator was not already applied. Shape [batch,Max_sources,num_hypothesis]
            dist_matrix = dist_matrix*large_mask # Shape [batch,Max_sources,num_hypothesis].
            
            # We then remove the hypothesis selected above (with minimum dist)
            mask_selected = torch.zeros_like(dist_matrix,dtype=bool) #Shape [batch,Max_sources,num_hypothesis]
            mask_selected.scatter_(2, idx_selected.unsqueeze(-1), 1) # idx_selected new shape: [batch,Max_sources,1]. 
            # The assignement mask_selected[i,j,idx_selected[i,j]]=1 is performed. 
            # Shape of mask_selected: [batch,Max_sources,num_hypothesis]
            
            # assert mask_selected.shape == (batch,Max_sources,num_hyps)
            
            mask_selected = ~mask_selected #Shape [batch,Max_sources,num_hypothesis], we keep only the hypothesis which are not the minimum.
            dist_matrix = dist_matrix * mask_selected #Shape [batch,Max_sources,num_hypothesis]
            
            # Finally, we compute the loss
            count_non_zeros_2 = torch.sum(dist_matrix!=0)

            if count_non_zeros_2 > 0 :
                epsilon_loss = torch.multiply(torch.sum(dist_matrix)/count_non_zeros_2, epsilon) #Scalar for each hyp
            else : 
                epsilon_loss = torch.tensor(0.)
            
            sum_losses = torch.add(sum_losses, epsilon_loss) # Loss for the unselected (i.e., not winners) hypothesis (epsilon weighted)
            sum_losses = torch.add(sum_losses, loss0) # Loss for the selected (i.e., the winners) hypothesis (1-epsilon weighted)
            sum_losses = torch.add(sum_losses, conf_weight*confidence_loss) # Loss for the confidence prediction. 
            
        elif mode == 'wta-top-n' and top_n > 1:
            
            dist_matrix = torch.multiply(dist_matrix, -1) # Shape [batch,Max_sources,num_hyps]
            top_k, indices = torch.topk(input=dist_matrix, k=top_n, dim=-1) #top_k of shape [batch,Max_sources,top_n], indices of shape [batch,Max_sources,top_n]
            dist_matrix_min = torch.multiply(top_k,-1) 

            if distance=='spherical':
                dist_matrix_euclidean = torch.multiply(dist_matrix_euclidean, -1) # Shape (batch,Max_sources,num_hyps) 
                top_k, _ = torch.topk(input=dist_matrix_euclidean, k=top_n, dim=-1) #top_k of shape [batch,Max_sources,top_n], indices of shape [batch,Max_sources,top_n]
                dist_matrix_min_euclidean = torch.multiply(top_k,-1) 
                mask = dist_matrix_min_euclidean <= filling_value/2
            else : 
                mask = dist_matrix_min <= filling_value/2 # We create a mask of shape [batch,Max_sources,top_n] for only selecting the actives sources, i.e. those which were not filled with fake values. 
            # assert mask[:,:,0].all() == mask[:,:,-1].all() # This mask related should be constant in the third dimension.
            
            dist_matrix_min = dist_matrix_min*mask # [batch,Max_sources,top_n], we select only the active sources.  
            # assert dist_matrix_min.shape == (batch,Max_sources,top_n)
            
            count_non_zeros = torch.sum(mask[:,:,0]!=0) # We count the number of entries (in the first two dimensions) for which the mask is different from zero. 
            
            ### Confidence management
            # Create tensors to index batch and Max_sources and top-n dimensions. 
            batch_indices = torch.arange(batch)[:, None, None].expand(-1, Max_sources,top_n) # Shape [batch, Max_sources,top_n]
            # We set the confidence of the selected hypothesis
            gt_conf_stacked_t[batch_indices[mask], indices[mask]] = 1 #Shape (batch,num_hyps)
            ###

            #####
            selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores 
            # to be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.
            
            for i in range(top_n):

                # assert count_non_zeros == torch.sum(mask[:,:,i]!=0) # We count the number of entries for which the mask is different from zero. 
            
                if count_non_zeros>0 : 
                    loss  = torch.multiply(torch.sum(dist_matrix_min[:, :, i])/count_non_zeros, 1.0)

                else :
                    loss = torch.tensor(0.) 

                sum_losses = torch.add(sum_losses, loss/top_n)

            if count_non_zeros>0 : 
              
                unselected_mask = ~selected_confidence_mask # [batch,num_hyps], mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.  

                if rejection_method=='uniform_negative' : 
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack([torch.multinomial(unselected_mask[b_idx].float(), number_unconfident, replacement=False) for b_idx in range(batch)]) #(batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(-1, number_unconfident) #(batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    gt_conf_stacked_t[batch_indices, unconfident_indices] = 0 #(Useless) Line added for the sake of completness. 

                elif rejection_method=='all' :

                    selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(conf_pred_stacked_t[selected_confidence_mask], gt_conf_stacked_t[selected_confidence_mask])
            
            else :
                confidence_loss = torch.tensor(0.)

            sum_losses = torch.add(sum_losses, conf_weight*confidence_loss)
            
        return sum_losses