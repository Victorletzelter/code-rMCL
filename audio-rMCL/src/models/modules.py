import abc
from src.data.data_handlers import TUTSoundEvents
from src.metrics import frame_recall, doa_error, oracle_doa_error, emd_metric
import numpy as np
import os
import pandas as pd
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import json
import sys

class AbstractLocalizationModule(ptl.LightningModule, abc.ABC):
    """Abstract class for localization modules from https://github.com/chrschy/adrenaline, the official code of the paper [B]. 
    The test_step function was adapted such that it both handle the baseline setting and the multi-hypothesis (MH) and multi-hypothesis with confidence (MH-CONF) settings.
    
    [B] Schymura, C., Ochiai, T., Delcroix, M., Kinoshita, K., Nakatani, T., Araki, S., & Kolossa, D. (2021, January). Exploiting attention-based sequence-to-sequence architectures for sound event localization. In 2020 28th European Signal Processing Conference (EUSIPCO) (pp. 231-235). IEEE.
"""
    def __init__(self, dataset_path: str, cv_fold_idx: int, hparams):
        super(AbstractLocalizationModule, self).__init__()

        self.dataset_path = dataset_path
        self.cv_fold_idx = cv_fold_idx

        self._hparams = hparams
        self.max_num_sources = hparams['max_num_sources']
        
        if 'max_num_overlapping_sources_test' in hparams :
            self.max_num_overlapping_sources_test = hparams['max_num_overlapping_sources_test']
            
        else : 
            self.max_num_overlapping_sources_test = self.max_num_sources
         
        self.loss_function = self.get_loss_function()
        
    @property
    def hparams(self):
        return self._hparams

    @abc.abstractmethod
    def forward(self,
                audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_loss_function(self) -> nn.Module:
        raise NotImplementedError

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams['learning_rate'], weight_decay=0.0)

        lr_lambda = lambda epoch: self.hparams['learning_rate'] * np.minimum(
            (epoch + 1) ** -0.5, (epoch + 1) * (self.hparams['num_epochs_warmup'] ** -1.5)
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [scheduler]

    def training_step(self,
                      batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                      batch_idx: int) -> Dict:
        predictions, targets = self._process_batch(batch)

        loss, _ = self.loss_function(predictions, targets)

        output = {'loss': loss}
        self.log_dict(output)

        return output

    def validation_step(self,
                        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                        batch_idx: int) -> Dict:
        with torch.no_grad():
            predictions, targets = self._process_batch(batch)
            loss, _ = self.loss_function(predictions, targets)

        output = {'val_loss': loss}
        self.log_dict(output)

        return output

    def validation_epoch_end(self,
                             outputs: list) -> None:
        average_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log_dict({'val_loss': average_loss, 'learning_rate': learning_rate})
        
        return average_loss

    def test_step(self,
                  batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                  batch_idx: int,
                  dataset_idx: int = 0,
                  custom_metrics: bool = None, #If True, we are in the MH or MH-CONF setting. Else, we are in the baseline setting.
                  dist_type_eval: bool = None) -> Dict:
        
        predictions, targets = self._process_batch(batch)

        if dist_type_eval is None :
            dist_type_eval = self.hparams['dist_type_eval']

        if custom_metrics is None :
            if 'MH' in self.hparams['name'] :
                custom_metrics = True

                if 'CONF' in self.hparams['name'] :
                    hyps_DOAs_pred_stacked, conf_stacked,_ = predictions #Shape ([batch,T,self.num_hypothesis,2],[batch,T,self.num_hypothesis,1])
                
                else :            
                    hyps_DOAs_pred_stacked, _ = predictions #Shape [batch,T,self.num_hypothesis,2]
           
            else : 
                custom_metrics = False

        if custom_metrics : #If we are in a multi-hypothesis setting (MH or MHCONF).

            if 'CONF' in self.hparams['name'] : # If we are in the MH-CONF setting.
                predictions = (hyps_DOAs_pred_stacked,conf_stacked)
                emd = emd_metric(predictions = predictions, targets = targets, conf_mode = True, distance = dist_type_eval, 
                                                  dataset_idx=dataset_idx,batch_idx=batch_idx, num_sources_per_sample_min=self.hparams['num_sources_per_sample_min'],num_sources_per_sample_max=self.hparams['num_sources_per_sample_max'])
                oracle = oracle_doa_error(predictions = predictions, targets = targets, distance = dist_type_eval,dataset_idx=dataset_idx,batch_idx=batch_idx,num_sources_per_sample_min=self.hparams['num_sources_per_sample_min'],num_sources_per_sample_max=self.hparams['num_sources_per_sample_max'])
                output = {'test_oracle_doa_error'+'_'+dist_type_eval[0:4]: oracle,
                    'test_emd_metric'+'_'+dist_type_eval[0:4]: emd[0],
                    'test_std_emd_metric'+'_'+dist_type_eval[0:4]: emd[1],
                    'test_frame_recall'+'_'+dist_type_eval[0:4]: frame_recall(predictions, targets, mh_mode=True, conf_mode=True)}

            else : #If we are in the MH setting. 
                emd = emd_metric(predictions = predictions, targets = targets, conf_mode = False, distance = dist_type_eval, 
                                                  dataset_idx=dataset_idx,batch_idx=batch_idx,num_sources_per_sample_min=self.hparams['num_sources_per_sample_min'],num_sources_per_sample_max=self.hparams['num_sources_per_sample_max'])
                oracle = oracle_doa_error(predictions = predictions, targets = targets, distance = dist_type_eval, dataset_idx=dataset_idx,batch_idx=batch_idx, num_sources_per_sample_min=self.hparams['num_sources_per_sample_min'],num_sources_per_sample_max=self.hparams['num_sources_per_sample_max'])
                output = {'test_oracle_doa_error'+'_'+dist_type_eval[0:4]: oracle,
                          'test_emd_metric'+'_'+dist_type_eval[0:4]: emd[0],
                          'test_std_emd_metric'+'_'+dist_type_eval[0:4]: emd[1]}
    
        else :
            predicted_sa = torch.sigmoid(predictions[0][:,:,:,np.newaxis]) > 0.5
            predictions_for_emd = (predictions[1],predicted_sa) # Switch because : DOA first then confidences                      
            emd = emd_metric(predictions = predictions_for_emd, targets = targets, conf_mode = True, distance = dist_type_eval, dataset_idx=dataset_idx,batch_idx=batch_idx,num_sources_per_sample_min=self.hparams['num_sources_per_sample_min'],num_sources_per_sample_max=self.hparams['num_sources_per_sample_max'])
            
            predictions_for_oracle = (predictions[1],predicted_sa)
            oracle = oracle_doa_error(predictions = predictions_for_oracle, targets = targets, distance = dist_type_eval, dataset_idx=dataset_idx,batch_idx=batch_idx, activity_mode=True,num_sources_per_sample_min=self.hparams['num_sources_per_sample_min'],num_sources_per_sample_max=self.hparams['num_sources_per_sample_max'])

            output = {'test_oracle_doa_error'+'_'+dist_type_eval[0:4]: oracle,
                'test_emd_metric'+'_'+dist_type_eval[0:4]: emd[0],
                'test_frame_recall': frame_recall(predictions, targets), 'test_doa_error': doa_error(predictions, targets)
            }
        self.log_dict(output)

        return output

    def test_epoch_end(self,
                       outputs: List, 
                       custom_metrics: bool = None,
                       dist_type_eval: bool = None) -> None:
        dataset_name = os.path.split(self.dataset_path)[-1]
        run_folder_name = sys.stdout.name.split('terminal_output.txt')[-2].split('/')[-2]
        
        if dist_type_eval is None :
            dist_type_eval = self.hparams['dist_type_eval']

        if custom_metrics is None :
            if 'MH' in self.hparams['name'] :
                custom_metrics = True
            else : 
                custom_metrics = False
        
        if custom_metrics is True :
 
            results = {
            'model': [], 'dataset': [], 'fold_idx': [], 'subset_idx': [], 
            'oracle_doa_error':[], 'average_oracle_doa_error':[], 'std_oracle_doa_error':[],
            'emd_metric' : [], 'average_emd_metric': [], 'std_emd_metric': []}
            
            num_subsets = len(outputs)

            for subset_idx in range(num_subsets):
                
                doa_error = torch.stack([x['test_oracle_doa_error'+'_'+dist_type_eval[0:4]] for x in outputs[subset_idx]]).detach().cpu().numpy()
                num_sequences = len(doa_error)

                emd_metric = torch.stack([x['test_emd_metric'+'_'+dist_type_eval[0:4]] for x in outputs[subset_idx]]).detach().cpu().numpy()
                _results = {'emd_metric' : [], 'oracle_doa_error':[]}

                for seq_idx in range(num_sequences):
                    _results['oracle_doa_error'].append(float(doa_error[seq_idx]))
                    _results['emd_metric'].append(float(emd_metric[seq_idx]))
                    
                data_frame = pd.DataFrame.from_dict(_results)       
                average_doa_error = torch.tensor(data_frame['oracle_doa_error'].mean(), dtype=torch.float32)
                std_doa_error = torch.tensor(data_frame['oracle_doa_error'].std(), dtype=torch.float32)
                
                results['model'].append(self.hparams['name'])
                results['dataset'].append(dataset_name)
                results['fold_idx'].append(self.cv_fold_idx)
                results['subset_idx'].append(subset_idx)
                results['average_oracle_doa_error'].append(float(average_doa_error))
                results['std_oracle_doa_error'].append(float(std_doa_error))
                results['oracle_doa_error'].append(_results['oracle_doa_error'])

                average_emd_metric = torch.tensor(data_frame['emd_metric'].mean(), dtype=torch.float32)
                std_emd_metric = torch.tensor(data_frame['emd_metric'].std(), dtype=torch.float32)
                results['average_emd_metric'].append(float(average_emd_metric))
                results['std_emd_metric'].append(float(std_emd_metric))
                results['emd_metric'].append(_results['emd_metric'])

            results_file = os.path.join(self.hparams['results_dir'], 'MH_'+self.hparams['name'] + 
                                        '_mode_'+
                                        str(self.hparams['mode'])+
                                        '_num_hyps_'+
                                        str(self.hparams['num_hypothesis'])+ '_'+
                                        dataset_name + '_fold' + str(self.cv_fold_idx) + '_'
                                        'max_sources'+ str(self.max_num_sources) +  '_' +
                                        'num_test_dataloders'+ str(num_subsets)+'-'+ run_folder_name+'-'+'emd'+
                                        str(np.round(float(average_emd_metric),3))+'.json')   
            
            # Result file overriden, else use if not os.path.isfile(results_file):
            if not os.path.isdir(self.hparams['results_dir']):
                os.makedirs(self.hparams['results_dir'])  

            with open(str(results_file),'w') as file : 
                json.dump(results, file)   
                 
            self.log_dict({'test_mean_emd_metric': np.nanmean(results['average_emd_metric']), 'test_mean_oracle_doa_error': np.nanmean(results['average_oracle_doa_error'])}) 
            
        elif custom_metrics is False : 

            results = {
            'model': [], 'dataset': [], 'fold_idx': [], 'subset_idx': [], 'frame_recall' : [],
            'doa_error' : [], 'average_frame_recall': [], 'average_doa_error': [],
            'std_frame_recall': [], 'std_doa_error': [],
            'oracle_doa_error' : [], 'average_oracle_doa_error': [], 'std_oracle_doa_error': [],
            'emd_metric' : [], 'average_emd_metric': [], 'std_emd_metric': []
        }
            
            num_subsets = len(outputs)

            for subset_idx in range(num_subsets):
                
                frame_recall = torch.stack([x['test_frame_recall'] for x in outputs[subset_idx]]).detach().cpu().numpy()
                doa_error = torch.stack([x['test_doa_error'] for x in outputs[subset_idx]]).detach().cpu().numpy()
                oracle_doa_error = torch.stack([x['test_oracle_doa_error'+'_'+dist_type_eval[0:4]] for x in outputs[subset_idx]]).detach().cpu().numpy()
                emd_metric = torch.stack([x['test_emd_metric'+'_'+dist_type_eval[0:4]] for x in outputs[subset_idx]]).detach().cpu().numpy()

                num_sequences = len(frame_recall)
                
                _results = {'frame_recall' : [], 'doa_error' : [], 'oracle_doa_error' : [], 'emd_metric' : []} 

                for seq_idx in range(num_sequences):
                    _results['frame_recall'].append(float(frame_recall[seq_idx]))
                    _results['doa_error'].append(float(doa_error[seq_idx]))
                    _results['oracle_doa_error'].append(float(oracle_doa_error[seq_idx]))
                    _results['emd_metric'].append(float(emd_metric[seq_idx])) 
                    
                data_frame = pd.DataFrame.from_dict(_results)
                
                average_frame_recall = torch.tensor(data_frame['frame_recall'].mean(), dtype=torch.float32)
                std_frame_recall = torch.tensor(data_frame['frame_recall'].std(), dtype=torch.float32)
                average_doa_error = torch.tensor(data_frame['doa_error'].mean(), dtype=torch.float32)
                std_doa_error = torch.tensor(data_frame['doa_error'].std(), dtype=torch.float32)
                average_oracle_doa_error = torch.tensor(data_frame['oracle_doa_error'].mean(), dtype=torch.float32)
                std_oracle_doa_error = torch.tensor(data_frame['oracle_doa_error'].std(), dtype=torch.float32)
                average_emd_metric = torch.tensor(data_frame['emd_metric'].mean(), dtype=torch.float32)
                std_emd_metric = torch.tensor(data_frame['emd_metric'].std(), dtype=torch.float32)
                
                results['model'].append(self.hparams['name'])
                results['dataset'].append(dataset_name)
                results['fold_idx'].append(self.cv_fold_idx)
                results['subset_idx'].append(subset_idx)
                results['average_frame_recall'].append(float(average_frame_recall))
                results['average_doa_error'].append(float(average_doa_error))
                results['std_frame_recall'].append(float(std_frame_recall))
                results['std_doa_error'].append(float(std_doa_error))
                results['frame_recall'].append(_results['frame_recall'])
                results['doa_error'].append(_results['doa_error'])
                results['oracle_doa_error'].append(_results['oracle_doa_error'])
                results['average_oracle_doa_error'].append(float(average_oracle_doa_error))
                results['std_oracle_doa_error'].append(float(std_oracle_doa_error))
                results['emd_metric'].append(_results['emd_metric'])
                results['average_emd_metric'].append(float(average_emd_metric))
                results['std_emd_metric'].append(float(std_emd_metric))

            results_file = os.path.join(self.hparams['results_dir'], self.hparams['name'] + '_'
                                        + dataset_name + '_' + 'fold' + str(self.cv_fold_idx) + '_'
                                    'max_sources'+ str(self.max_num_sources) +  '_' +
                                    'num_test_dataloders'+ str(num_subsets) + '-'+ run_folder_name+'-'
                                    + 'emd'+str(np.round(results['average_emd_metric'][-1],3))+'.json') 

            if not os.path.isdir(self.hparams['results_dir']):
                os.makedirs(self.hparams['results_dir'])  
                    
            with open(str(results_file),'w') as file : 
                json.dump(results, file)    

            self.log_dict({'test_frame_recall': average_frame_recall, 'test_doa_error': average_doa_error})

    def _process_batch(self,
                       batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                       ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        audio_features, targets = batch
        predictions = self.forward(audio_features)

        return predictions, targets

    def train_dataloader(self) -> DataLoader:
        train_dataset = TUTSoundEvents(self.dataset_path, split='train',
                                       tmp_dir=self.hparams['tmp_dir'],
                                       test_fold_idx=self.cv_fold_idx,
                                       sequence_duration=self.hparams['sequence_duration'],
                                       chunk_length=self.hparams['chunk_length'],
                                       frame_length=self.hparams['frame_length'],
                                       num_fft_bins=self.hparams['num_fft_bins'],
                                       max_num_sources=self.hparams['max_num_sources'])

        return DataLoader(train_dataset, shuffle=True, batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'])

    def val_dataloader(self) -> DataLoader:
        valid_dataset = TUTSoundEvents(self.dataset_path, split='valid',
                                       tmp_dir=self.hparams['tmp_dir'],
                                       test_fold_idx=self.cv_fold_idx,
                                       sequence_duration=self.hparams['sequence_duration'],
                                       chunk_length=self.hparams['chunk_length'],
                                       frame_length=self.hparams['frame_length'],
                                       num_fft_bins=self.hparams['num_fft_bins'],
                                       max_num_sources=self.hparams['max_num_sources'])

        return DataLoader(valid_dataset, shuffle=False, batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'])

    def test_dataloader(self) -> List[DataLoader]:
        # During testing, a whole sequence is packed into one batch. The batch size set for training and validation
        # is ignored in this case.
        num_chunks_per_sequence = int(self.hparams['sequence_duration'] / self.hparams['chunk_length'])

        test_loaders = []

        for num_overlapping_sources in range(1, min(self.max_num_overlapping_sources_test,3)+1):
            test_dataset = TUTSoundEvents(self.dataset_path, split='test',
                                          tmp_dir=self.hparams['tmp_dir'],
                                          test_fold_idx=self.cv_fold_idx,
                                          sequence_duration=self.hparams['sequence_duration'],
                                          chunk_length=self.hparams['chunk_length'],
                                          frame_length=self.hparams['frame_length'],
                                          num_fft_bins=self.hparams['num_fft_bins'],
                                          max_num_sources=self.hparams['max_num_sources'],
                                          num_overlapping_sources=num_overlapping_sources)

            test_loaders.append(DataLoader(test_dataset, shuffle=False, batch_size=num_chunks_per_sequence,
                                           num_workers=self.hparams['num_workers']))

        return test_loaders

class FeatureExtraction(nn.Module):
    """CNN-based feature extraction originally proposed in [1].

    Args:
        num_steps_per_chunk: Number of time steps per chunk, which is required for correct layer normalization.
        num_fft_bins: Number of FFT bins used for spectrogram computation.
        dropout_rate: Dropout rate.

    References:
        [1] Sharath Adavanne, Archontis Politis, Joonas Nikunen, and Tuomas Virtanen, "Sound event localization and
            detection of overlapping sources using convolutional recurrent neural network" in IEEE Journal of Selected
            Topics in Signal Processing (JSTSP 2018)
    """
    def __init__(self,
                 num_steps_per_chunk: int,
                 num_fft_bins: int,
                 dropout_rate: float = 0.0) -> None:
        """Initialization of CNNs-based layers for features extraction. 

        Args:
            num_steps_per_chunk (int): Number of steps in each chunk.  
            num_fft_bins (int): Number of frequencies calculated at each FFT computation.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(FeatureExtraction, self).__init__()
        # As the number of audio channels in the raw data is four, this number doubles after frequency features extraction (amplitude and phase). 
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), ceil_mode=True),
            nn.Dropout(p=dropout_rate)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), ceil_mode=True),
            nn.Dropout(p=dropout_rate)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), ceil_mode=True),
            nn.Dropout(p=dropout_rate)
        )

        self.layer_norm = nn.LayerNorm([num_steps_per_chunk, int(num_fft_bins / 4)]) # Layer normalization used 
        # Statistics are calculated over the last two dimensions of the input.s

    def forward(self,
                audio_features: torch.Tensor) -> torch.Tensor:
        """Feature extraction forward pass.

        Args:
            audio_features (torch.Tensor): Input tensor with dimensions [batch,2*C,T,B], where batch is the batch size, T is the number of
        time steps per chunk, B is the number of FFT bins and C is the number of audio channels.

        Returns:
            torch.Tensor: Extracted features with dimension [batch,T,B/4].
        """
        output = self.conv_layer1(audio_features) # Output shape [batch,64,T,B/8]
        output = self.conv_layer2(output) # Output shape [batch,64,T,B/64]      
        output = self.conv_layer3(output) # Output shape [batch,64,T,B/256] 
        output = output.permute(0, 2, 1, 3)   # Output shape [batch,T,64,B/256]  
        batch_size, num_frames, _, _ = output.shape
        output = output.contiguous().view(batch_size, num_frames, -1) # Output shape [batch,T,B/4]

        return self.layer_norm(output)

class LocalizationOutput(nn.Module):
    """Implements a module that outputs source activity and direction-of-arrival for sound event localization. An input
    of fixed dimension is passed through a fully-connected layer and then split into a source activity vector with
    sigmoid output activations and corresponding azimuth and elevation vectors, which are subsequently combined to a
    direction-of-arrival output tensor. Credits to https://github.com/chrschy/adrenaline for the implementation.

    [B] Schymura, C., Ochiai, T., Delcroix, M., Kinoshita, K., Nakatani, T., Araki, S., & Kolossa, D. (2021, January). Exploiting attention-based sequence-to-sequence architectures for sound event localization. In 2020 28th European Signal Processing Conference (EUSIPCO) (pp. 231-235). IEEE.

    Args:
        input_dim: Input dimension.

        max_num_sources: Maximum number of sound sources that should be represented by the module.
    """
    def __init__(self, input_dim: int, max_num_sources: int):
        super(LocalizationOutput, self).__init__()

        self.source_activity_output = nn.Linear(input_dim, max_num_sources)
        self.azimuth_output = nn.Linear(input_dim, max_num_sources)
        self.elevation_output = nn.Linear(input_dim, max_num_sources)

    def forward(self,
                input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward pass.

        :param input: Input tensor with dimensions [batch,T,D], where batch is the batch size, T is the number of time steps per
                      chunk and D is the input dimension.
        :return: Tuple containing the source activity tensor of size [batch,T,S] and the direction-of-arrival tensor with
                 dimensions [batch,T,S,2], where S is the maximum number of sources.
        """
        source_activity = self.source_activity_output(input)

        azimuth = self.azimuth_output(input)
        elevation = self.elevation_output(input)
        direction_of_arrival = torch.cat((azimuth.unsqueeze(-1), elevation.unsqueeze(-1)), dim=-1)

        return source_activity, direction_of_arrival

### MH for multi-hypothesis.
class MHLocalizationOutput(nn.Module):
    """Implement a module that outputs multiple hypothesis for direction-of-arrival for sound source localization over time,
    given a feature vector for each time step as input (see the hypothesis-based.py file).
    Args:
        input_dim: Input dimension at each time step.    
        num_hypothesis: Number of hypothesis in the model.
    """
    def __init__(self, input_dim: int, num_hypothesis: int, output_dim: int=2):
        super(MHLocalizationOutput, self).__init__()

        self.source_activity_output_layers = {}
        self.azimuth_output_layers = {}
        self.elevation_output_layers = {}
        self.num_hypothesis = num_hypothesis
        self.output_dim = output_dim
        
        self.doa_layers = nn.ModuleDict()
        
        for k in range(self.num_hypothesis) :  
            self.doa_layers['hyp_'+'{}'.format(k)] = nn.Linear(in_features=input_dim, out_features=output_dim,device='cuda:0')
    
    def forward(self,
                input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward pass.

        :param input: Input tensor with dimensions [batch,T,D], where batch is the batch size, T is the number of time steps per
                    chunk and D is the input dimension.
        :return: Stacked direciton of arrival hypothesis with shape [batch,T,self.num_hypothesis,output_dim]
        """  
        directions_of_arrival = []
        batch, T = input.shape[0], input.shape[1]
        input_reshaped = input.reshape(-1,input.shape[-1]) #Of shape [batch*T x input_dim]
        
        for k in range(self.num_hypothesis) :
            directions_of_arrival.append((self.doa_layers['hyp_'+'{}'.format(k)](input_reshaped)).reshape(batch,T,-1)) # Size [batch,T,output_dim]
            
        hyp_stacked = torch.stack(directions_of_arrival, dim=-2) #Shape [batch,T,self.num_hypothesisxoutput_dim]

        return hyp_stacked

### MHCONF for multi-hypothesis and confidence based
class MHCONFLocalizationOutput(nn.Module):
    """Implement a module that outputs multiple hypothesis for direction-of-arrival and associated confidences scores for sound source localization over time,
    given a feature vector for each time step as input (see the hypothesis-confidence-based.py file).
    Args:
        input_dim: Input dimension at each time step.    
        num_hypothesis: Number of hypothesis in the model.
    """
    def __init__(self, input_dim: int, num_hypothesis: int, output_dim: int=2):
        super(MHCONFLocalizationOutput, self).__init__()

        self.source_activity_output_layers = {}
        self.azimuth_output_layers = {}
        self.elevation_output_layers = {}
        self.num_hypothesis = num_hypothesis
        self.output_dim = output_dim
        
        self.doa_layers = nn.ModuleDict()
        self.doa_conf_layers = nn.ModuleDict()
        
        for k in range(self.num_hypothesis) :  
            self.doa_layers['hyp_'+'{}'.format(k)] = nn.Linear(in_features=input_dim, out_features=output_dim,device='cuda:0')
            self.doa_conf_layers['hyp_'+'{}'.format(k)] = nn.Linear(in_features=input_dim, out_features=1,device='cuda:0')
           
    def forward(self,
                input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward pass.

        :param input: Input tensor with dimensions [batch,T,D], where batch is the batch size, T is the number of time steps per
                    chunk and D is the input dimension.
        :return: 
            Stacked direction of arrival hypothesis with shape [batch,T,self.num_hypothesisxoutput_dim]
            Stacked scores values for each hypothesis with shape [batch,T,self.num_hypothesisx1]
        """  
        directions_of_arrival = []
        associated_confidences = []
        batch, T = input.shape[0], input.shape[1]
        input_reshaped = input.reshape(-1,input.shape[-1]) #Of shape [batch*T , input_dim]
        
        for k in range(self.num_hypothesis) :
            directions_of_arrival.append((self.doa_layers['hyp_'+'{}'.format(k)](input_reshaped)).reshape(batch,T,-1)) # Size [batch,T,output_dim]
            associated_confidences.append((torch.nn.Sigmoid()(self.doa_conf_layers['hyp_'+'{}'.format(k)](input_reshaped))).reshape(batch,T,-1)) # Size [batch,T,1]

        hyp_stacked = torch.stack(directions_of_arrival, dim=-2) #Shape [batch,T,self.num_hypothesisxoutput_dim]
        conf_stacked = torch.stack(associated_confidences, dim=-2) #Shape [batch,T,self.num_hypothesisx1]

        return hyp_stacked, conf_stacked