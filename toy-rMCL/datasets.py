import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

class ToyDataset(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples):
        super(ToyDataset, self).__init__()
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        # Select a section according to the probabilities defined in the paper
        section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

        # Sample a point uniformly from the selected section
        if section == 1:
            x = np.random.uniform(-1, 0)
            y = np.random.uniform(-1, 0)
        elif section == 2:
            x = np.random.uniform(-1, 0)
            y = np.random.uniform(0, 1)
        elif section == 3:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(-1, 0)
        else:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        s1 = (-1, 0, -1, 0)
        s2 = (-1, 0, 0, 1)
        s3 = (0, 1, -1, 0)
        s4 = (0, 1, 0, 1)
        
        # Define the probabilities of selecting each section
        p1 = p4 = (1 - t) / 2
        p2 = p3 = t / 2
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):
            # Select a section
            section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
            
            # Sample a point uniformly from the selected section
            if section == 1:
                x = np.random.uniform(s1[0], s1[1])
                y = np.random.uniform(s1[2], s1[3])
            elif section == 2:
                x = np.random.uniform(s2[0], s2[1])
                y = np.random.uniform(s2[2], s2[3])
            elif section == 3:
                x = np.random.uniform(s3[0], s3[1])
                y = np.random.uniform(s3[2], s3[3])
            elif section == 4:
                x = np.random.uniform(s4[0], s4[1])
                y = np.random.uniform(s4[2], s4[3])
            else:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
                
            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSourcesToyDataset(data.Dataset):
    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples, Max_sources=2,grid_t=False,t=None) :
        super(MultiSourcesToyDataset, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        if t<=0.5:
            u = np.random.uniform(0,1)
            if u <= 1-2*t:
                N_sources = 2
            else : 
                N_sources = 1
        else :
            u = np.random.uniform(0,1)
            if u <= -1+2*t:
                N_sources = 2
            else : 
                N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper
            section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

            # Sample a point uniformly from the selected section
            if section == 1:
                output[source,0] = np.random.uniform(-1, 0)
                output[source,1] = np.random.uniform(-1, 0)
            elif section == 2:
                output[source,0] = np.random.uniform(-1, 0)
                output[source,1] = np.random.uniform(0, 1)
            elif section == 3:
                output[source,0] = np.random.uniform(0, 1)
                output[source,1] = np.random.uniform(-1, 0)
            else:
                output[source,0] = np.random.uniform(0, 1)
                output[source,1] = np.random.uniform(0, 1)

        return t, output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            if t<=0.5:
                u = np.random.uniform(0,1)
                if u <= 1-2*t:
                    N_sources = 2
                else : 
                    N_sources = 1
            else :
                u = np.random.uniform(0,1)
                if u <= -1+2*t:
                    N_sources = 2
                else : 
                    N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper
                section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

                # Sample a point uniformly from the selected section
                if section == 1:
                    output[source,0] = np.random.uniform(-1, 0)
                    output[source,1] = np.random.uniform(-1, 0)
                elif section == 2:
                    output[source,0] = np.random.uniform(-1, 0)
                    output[source,1] = np.random.uniform(0, 1)
                elif section == 3:
                    output[source,0] = np.random.uniform(0, 1)
                    output[source,1] = np.random.uniform(-1, 0)
                else:
                    output[source,0] = np.random.uniform(0, 1)
                    output[source,1] = np.random.uniform(0, 1)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity