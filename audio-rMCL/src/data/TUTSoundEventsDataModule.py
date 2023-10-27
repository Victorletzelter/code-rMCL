import pytorch_lightning as pl
from src.data.data_handlers.tut_sound_events import TUTSoundEvents
from torch.utils.data import DataLoader

class TUTSoundEventsDataModule(pl.LightningDataModule):
    def __init__(self, root: str,
                 tmp_dir: str = './tmp',
                 test_fold_idx: int = 1,
                 sequence_duration: float = 30.,
                 chunk_length: float = 0.5,
                 frame_length: float = 0.04,
                 num_fft_bins: int = 2048,
                 max_num_sources: int = 5,
                 num_overlapping_sources: int = None,
                 batch_size: int = 32,
                 num_workers: int = 16,
                 **kwargs):
        """Data module associated with the TUT Sound Event Datasets.

        Args:
            root (str): Dataset folder absolute path. 
            tmp_dir (str, optional): Folder for storing temporary files, i.e., the audio features and labels in each chunks. Defaults to './tmp'.
            test_fold_idx (int, optional): Choice of the cross-validation folder, i.e., the index of the folder (1,2 or 3) on which the testing will be performed. Defaults to 1.
            sequence_duration (float, optional): Duration (s) of the audio files. If else, padding or cropping performed. Defaults to 30.
            chunk_length (float, optional): Duration (s) of the non-overlapping "chunks". Defaults to 2.
            frame_length (float, optional): Length (s) of the analysis window (Default to hann) used for FFT computation. Defaults to 0.04.
            num_fft_bins (int, optional):  Number of frequencies calculated at each FFT computation. Defaults to 2048.
            max_num_sources (int, optional): Maximum number of sources in the model output. The data will be formatted according to this value. Defaults to 5.
            num_overlapping_sources (int, optional): Refers to the choice of the dataset according to the maximum number of overlapping sources (1,2 or 3). Defaults to None.
            batch_size (int, optional): Number of samples to load in each batch. Defaults to 32.
            num_workers (int, optional): Number of worker threads for multi-process data loading. Defaults to 16.
        """

        super().__init__()
        self.root = root
        self.tmp_dir = tmp_dir
        self.test_fold_idx = test_fold_idx
        self.sequence_duration = sequence_duration
        self.chunk_length = chunk_length
        self.frame_length = frame_length
        self.num_fft_bins = num_fft_bins
        self.max_num_sources = max_num_sources
        self.num_overlapping_sources = num_overlapping_sources
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if 'max_num_overlapping_sources_test' in kwargs :
            self.max_num_overlapping_sources_test = kwargs['max_num_overlapping_sources_test']
            
        else : 
            self.max_num_overlapping_sources_test = self.max_num_sources

    def setup(self, stage=None):
        # Define dataset split for train/val/test
        self.train_dataset = TUTSoundEvents(self.root,  tmp_dir = self.tmp_dir, split='train', test_fold_idx=self.test_fold_idx,
                                            sequence_duration=self.sequence_duration, chunk_length=self.chunk_length,
                                            frame_length=self.frame_length, num_fft_bins=self.num_fft_bins,
                                            max_num_sources=self.max_num_sources, 
                                            num_overlapping_sources=self.num_overlapping_sources,
                                            )
        self.val_dataset = TUTSoundEvents(self.root,  tmp_dir = self.tmp_dir, split='valid', test_fold_idx=self.test_fold_idx,
                                          sequence_duration=self.sequence_duration, chunk_length=self.chunk_length,
                                          frame_length=self.frame_length, num_fft_bins=self.num_fft_bins,
                                          max_num_sources=self.max_num_sources,
                                          num_overlapping_sources=self.num_overlapping_sources,
                                          )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self) :
        # During testing, a whole sequence is packed into one batch. The batch size set for training and validation
        # is ignored in this case.
        num_chunks_per_sequence = int(self.sequence_duration / self.chunk_length)

        test_loaders = []

        for num_overlapping_sources in range(1, min(self.max_num_overlapping_sources_test,3)+1):
            test_dataset = TUTSoundEvents(self.root,  tmp_dir = self.tmp_dir, split='test', test_fold_idx=self.test_fold_idx,
                                        sequence_duration=self.sequence_duration, chunk_length=self.chunk_length,
                                        frame_length=self.frame_length, num_fft_bins=self.num_fft_bins,
                                        max_num_sources=self.max_num_sources,
                                        num_overlapping_sources=num_overlapping_sources)

            test_loaders.append(DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False))

        return test_loaders
