from argparse import Namespace
from .modules import AbstractLocalizationModule, FeatureExtraction, MHLocalizationOutput
import torch
import torch.nn as nn
from typing import Tuple
from src.utils.utils import wta_loss

class MHSELDNet(AbstractLocalizationModule):
    """Custom multi-hypothesis version adapted from the seldnet architecture (see the seldnet.py file)."""
    def __init__(self,
                 dataset_path: str,
                 cv_fold_idx: int,
                 hparams: Namespace) -> None:
        super(MHSELDNet, self).__init__(dataset_path, cv_fold_idx, hparams)

        num_steps_per_chunk = int(2 * hparams['chunk_length'] / hparams['frame_length'])
        self.feature_extraction = FeatureExtraction(num_steps_per_chunk,
                                                    hparams['num_fft_bins'],
                                                    dropout_rate=hparams['dropout_rate'])

        feature_dim = int(hparams['num_fft_bins'] / 4) # See the FeatureExtraction module for the justification of this 
        # value for the feature_dim. 
        
        self.gru = nn.GRU(feature_dim, hparams['hidden_dim'], num_layers=4, batch_first=True, bidirectional=True)

        self.localization_output = MHLocalizationOutput(input_dim =2 * hparams['hidden_dim'], 
                                                        num_hypothesis = hparams['num_hypothesis'])
        # In the localization module, the input_dim is to 2 * hparams.hidden_dim if bidirectional=True in the GRU.

    def get_loss_function(self) -> nn.Module:
        return wta_loss(mode=self.hparams['mode'], top_n = self.hparams['top_n'],distance = self.hparams['distance'],
                         epsilon=self.hparams['epsilon'],single_target_loss=self.hparams['single_target_loss'])

    def forward(self,
                audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        extracted_features = self.feature_extraction(audio_features) # extracted_features of shape
        #[batch,T,B/4] where batch is the batch size, T is the number of time steps per chunk, and B is the number of FFT bins.

        output, _ = self.gru(extracted_features) # output of shape [batch,T,hparams['hidden_dim']]

        MHdirection_of_arrival_output = self.localization_output(output) # output of shape [batch,T,num_hypothesis,2]
        meta_data = {}

        return MHdirection_of_arrival_output, meta_data
