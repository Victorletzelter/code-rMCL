import hashlib
import json
import numpy as np
import os
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft
from torch.utils.data import Dataset
from typing import Tuple
import h5py

# File copied and adapted from (https://github.com/chrschy/adrenaline/blob/master/data_handlers/tut_sound_events.py) [A]
# 
# [A] Schymura, C., Ochiai, T., Delcroix, M., Kinoshita, K., Nakatani, T., Araki, S., & Kolossa, D. (2021, January). Exploiting attention-based sequence-to-sequence architectures for sound event localization. 
# In 2020 28th European Signal Processing Conference (EUSIPCO) (pp. 231-235). IEEE.

class TUTSoundEvents(Dataset):
    """
    This class enables using a subset of the Tampere University of Technology Sound Events datasets. For more
    detailed information about these datasets, please refer to

        https://github.com/sharathadavanne/seld-net

    The following subsets are currently supported by this class:

        - ANSIM: Ambisonic, Anechoic and Synthetic Impulse Response Dataset (https://doi.org/10.5281/zenodo.1237703)
        - RESIM: Ambisonic, Reverberant and Synthetic Impulse Response Dataset (https://doi.org/10.5281/zenodo.1237707)
        - REAL: Ambisonic, Reverberant and Real-life Impulse Response Dataset (https://doi.org/10.5281/zenodo.1237793)

    Please run the script download_data.sh first to download these subsets from the corresponding repositories.
    """
    def __init__(self, root: str,
                 tmp_dir: str = './tmp',
                 split: str = 'train',
                 test_fold_idx: int = 1,
                 sequence_duration: float = 30.,
                 chunk_length: float = 0.5,
                 frame_length: float = 0.04,
                 num_fft_bins: int = 2048,
                 max_num_sources: int = 3,
                 num_overlapping_sources: int = None,
                 **kwargs) -> None:
        """Class constructor.

        :param root: path to root directory of the desired subset
        :param split: choose between 'train' (default), 'valid' and 'test'
        :param test_fold_idx: cross-validation index used for testing; choose between 1, 2, and 3
        :param sequence_duration: fixed duration of each audio signal in seconds, which is set to 30s by default
        :param num_fft_bins: number of FFT bins
        """
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
        self.tmp_dir = tmp_dir

        if split not in ['train', 'valid', 'test']:
            raise RuntimeError('Split must be specified as either train, valid or test.')

        if (test_fold_idx < 1) or (test_fold_idx > 3):
            raise RuntimeError('The desired test fold index must be specified as either 1, 2 or 3.')

        self.split = split
        self.test_fold_idx = test_fold_idx
        self.sequence_duration = sequence_duration
        self.chunk_length = chunk_length
        self.num_chunks_per_sequence = int(self.sequence_duration / self.chunk_length)
        self.frame_length = frame_length
        self.num_fft_bins = num_fft_bins
        self.max_num_sources = max_num_sources
        self.num_overlapping_sources = num_overlapping_sources

        # Assemble table containing paths to all audio and annotation files.
        self.sequences = {}

        for audio_subfolder in os.listdir(root):
            if os.path.isdir(os.path.join(root, audio_subfolder)) and audio_subfolder.startswith('wav'):
                annotation_subfolder = 'desc' + audio_subfolder[3:-5]

                if num_overlapping_sources is not None:
                    if num_overlapping_sources != int(annotation_subfolder[annotation_subfolder.find('ov') + 2]):
                        continue

                fold_idx = int(annotation_subfolder[annotation_subfolder.find('split') + 5])

                for file in os.listdir(os.path.join(root, audio_subfolder)):
                    file_prefix, extension = os.path.splitext(file)

                    if extension == '.wav':
                        path_to_audio_file = os.path.join(root, audio_subfolder, file)
                        path_to_annotation_file = os.path.join(root, annotation_subfolder, file_prefix + '.csv')
                        is_train_file = file_prefix.startswith('train')

                        # Check all three possible cases where files will be added to the global file list
                        if (split == 'train') and (fold_idx != test_fold_idx) and is_train_file:
                            self._append_sequence(path_to_audio_file, path_to_annotation_file, is_train_file, fold_idx, num_overlapping_sources)
                        elif (split == 'valid') and (fold_idx != test_fold_idx) and not is_train_file:
                            self._append_sequence(path_to_audio_file, path_to_annotation_file, is_train_file, fold_idx, num_overlapping_sources)
                        elif (split == 'test') and (fold_idx == test_fold_idx):
                            self._append_sequence(path_to_audio_file, path_to_annotation_file, is_train_file, fold_idx, num_overlapping_sources)

    def _append_sequence(self,
                         audio_file: str,
                         annotation_file: str,
                         is_train_file: bool,
                         fold_idx: int,
                         num_overlapping_sources: int) -> None:
        """Appends sequence (audio and annotation file) to global list of sequences.

        :param audio_file: path to audio file
        :param annotation_file: path to corresponding annotation file in *.csv-format
        :param is_train_file: flag indicating if file is used for training
        :param fold_idx: cross-validation fold index of current file
        :param num_overlapping_sources: number of overlapping sources in the dataset
        """
        for chunk_idx in range(self.num_chunks_per_sequence):
            sequence_idx = len(self.sequences)

            start_time = chunk_idx * self.chunk_length
            end_time = start_time + self.chunk_length

            self.sequences[sequence_idx] = {
                'audio_file': audio_file, 'annotation_file': annotation_file, 'is_train_file': is_train_file,
                'cv_fold_idx': fold_idx, 'chunk_idx': chunk_idx, 'start_time': start_time, 'end_time': end_time,
                'num_overlapping_sources': num_overlapping_sources
            }

    def _get_audio_features(self,
                            audio_file: str,
                            start_time: float = None,
                            end_time: float = None) -> np.ndarray:
        """Returns magnitude and phase of the multi-channel spectrogram for a given audio file.

        :param audio_file: path to audio file
        :param start_time: start time of the desired chunk in seconds
        :param end_time: end time of the desired chunk in seconds
        :return: magnitude, phase and sampling rate in Hz
        """
        sampling_rate, audio_data = wavfile.read(audio_file)
        num_samples, num_channels = audio_data.shape

        required_num_samples = int(sampling_rate * self.sequence_duration)

        # Perform zero-padding (if required) or truncate signal if it exceeds the desired duration.
        if num_samples < required_num_samples:
            audio_data = np.pad(audio_data, ((0, required_num_samples - num_samples), (0, 0)), mode='constant')
        elif num_samples > required_num_samples:
            audio_data = audio_data[:required_num_samples, :]

        # Normalize and crop waveform
        start_time_samples = int(start_time * sampling_rate)
        end_time_samples = int(end_time * sampling_rate)

        waveform = audio_data[start_time_samples:end_time_samples, :]
        waveform = waveform / np.iinfo(waveform.dtype).max

        # Compute multi-channel STFT and remove first coefficient and last frame
        frame_length_samples = int(self.frame_length * sampling_rate)
        spectrogram = stft(waveform, fs=sampling_rate, nperseg=frame_length_samples, nfft=self.num_fft_bins, axis=0)[-1]
        spectrogram = spectrogram[1:, :, :-1]
        spectrogram = np.transpose(spectrogram, [1, 2, 0])

        # Compose output tensor as concatenated magnitude and phase spectra
        audio_features = np.concatenate((np.abs(spectrogram), np.angle(spectrogram)), axis=0)

        return audio_features.astype(np.float16)

    def _get_targets(self,
                     annotation_file: str,
                     chunk_start_time: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a polar map of directions-of-arrival (azimuth and elevation) from a given annotation file.

        :param annotation_file: path to annotation file
        :param chunk_start_time: start time of the desired chunk in seconds
        :return: two-dimensional DoA map
        """

        annotations = pd.read_csv(annotation_file, header=0, names=[
            'sound_event_recording', 'start_time', 'end_time', 'elevation', 'azimuth', 'distance'])
        annotations = annotations.sort_values('start_time')

        event_start_time = annotations['start_time'].to_numpy()
        event_end_time = annotations['end_time'].to_numpy()

        num_frames_per_chunk = int(2 * self.chunk_length / self.frame_length) # The number of "frames" per chunk (50% overlap)

        source_activity = np.zeros((num_frames_per_chunk, self.max_num_sources), dtype=np.uint8)
        direction_of_arrival = np.zeros((num_frames_per_chunk, self.max_num_sources, 2), dtype=np.float32)

        for frame_idx in range(num_frames_per_chunk):
            frame_start_time = chunk_start_time + frame_idx * (self.frame_length / 2) # Compute the start time of the frame
            frame_end_time = frame_start_time + (self.frame_length / 2) # Compute the end time of the frame (frame with 50% overlap)

            event_mask = event_start_time <= frame_start_time # The event has started before the frame
            event_mask = event_mask | ((event_start_time >= frame_start_time) & (event_start_time < frame_end_time)) # The event has started before the frame or during the frame
            event_mask = event_mask & (event_end_time > frame_start_time) # The event has not ended before the frame

            events_in_chunk = annotations[event_mask] # Get the events that are in the frame
            num_active_sources = len(events_in_chunk) # Get the number of active sources in the frame (i.e., the number of events in the frame)

            if num_active_sources > 0: # If there are active sources in the frame
                source_activity[frame_idx, :num_active_sources] = 1 # Set the source activity to 1
                
                direction_of_arrival[frame_idx, :num_active_sources, :] = np.deg2rad(
                    events_in_chunk[['azimuth', 'elevation']].to_numpy()) # Extract the DOAs of the active sources

        return source_activity, direction_of_arrival

    def _get_parameter_hash(self) -> str:
        """Returns a hash value encoding the dataset parameter settings.

        :return: hash value
        """
        parameter_dict = {
            'chunk_length': self.chunk_length, 'frame_length': self.frame_length, 'num_fft_bins': self.num_fft_bins,
            'sequence_duration': self.sequence_duration
        }

        return hashlib.md5(json.dumps(parameter_dict, sort_keys=True).encode('utf-8')).hexdigest()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self,
                    index: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        sequence = self.sequences[index]

        file_path, file_name = os.path.split(sequence['audio_file'])
        group_path, group_name = os.path.split(file_path)
        _, dataset_name = os.path.split(group_path)
        parameter_hash = self._get_parameter_hash()

        feature_file_name = file_name + '_' + str(sequence['chunk_idx']) + '_f.h5'
        target_file_name = file_name + '_' + str(sequence['chunk_idx']) + '_t' + str(self.max_num_sources) + '.h5'

        path_to_feature_file = os.path.join(self.tmp_dir, dataset_name, group_name, parameter_hash)
        if not os.path.isdir(path_to_feature_file):
            try:
                os.makedirs(path_to_feature_file)
            except:
                pass
            
        if os.path.isfile(os.path.join(path_to_feature_file, feature_file_name)):
            with h5py.File(os.path.join(path_to_feature_file, feature_file_name), 'r') as f:
                audio_features = f['audio_features'][:]
        else:
            audio_features = self._get_audio_features(sequence['audio_file'], sequence['start_time'], sequence['end_time'])
            with h5py.File(os.path.join(path_to_feature_file, feature_file_name), 'w') as f:
                f.create_dataset('audio_features', data=audio_features, compression='gzip')

        if os.path.isfile(os.path.join(path_to_feature_file, target_file_name)):
            with h5py.File(os.path.join(path_to_feature_file, target_file_name), 'r') as f:
                source_activity = f['source_activity'][:]
                direction_of_arrival = f['direction_of_arrival'][:]
        else:
            source_activity, direction_of_arrival = self._get_targets(sequence['annotation_file'], sequence['start_time'])
            with h5py.File(os.path.join(path_to_feature_file, target_file_name), 'w') as f:
                f.create_dataset('source_activity', data=source_activity, compression='gzip')
                f.create_dataset('direction_of_arrival', data=direction_of_arrival, compression='gzip')

        return audio_features.astype(np.float32), (source_activity.astype(np.float32), direction_of_arrival.astype(np.float32))
