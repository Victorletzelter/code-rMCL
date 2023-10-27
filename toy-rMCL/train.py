import argparse
import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from IPython.display import display
from torch.utils.data import DataLoader

from datasets import MultiSourcesToyDataset
from losses import rmcl_loss, mhloss
from models import rmcl, smcl
from utils import set_seed, save_checkpoint, keep_best_checkpoint, load_and_merge_configs

def train_model(name,
                model, 
                criterion, 
                optimizer, 
                train_loader, 
                val_loader, 
                num_epochs=10, 
                device=torch.device("cpu"),
                save_freq=10,
                checkpoint_directory="checkpoint_directory",
                batch_size=1024,
                plot_losses=False,
                patience=20,
                seed=1234,
                ensemble_mode=False):
    """Script for training a model.

    Args:
        name (str): Name of the model to train. 
        model: Model to train.
        criterion: Loss used. 
        optimizer: Optimizer used. 
        train_loader: Training data loader.
        val_loader: Validation data loader.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        device (_type_, optional): Device used. Defaults to torch.device("cpu").
        save_freq (int, optional): Checkpoint save frequency. Defaults to 10.
        checkpoint_directory (str, optional): Checkpoint directory. Defaults to "checkpoint_directory".
        batch_size (int, optional): Batch size used. Defaults to 1024.
        plot_losses (bool, optional): Whether to plot the training and validation losses after training. Defaults to False.
        patience (int, optional): Patience parameter to early stopping. Defaults to 20.
        seed (int, optional): Seed used. Defaults to 1234.
        ensemble_mode (bool, optional): Whether to performe the training of the ensemble. Defaults to False.
    """

    train_loss = []
    train_loss_epochs = []
    val_loss = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Train the model
        train_loss_in_epoch= []

        for _, data in enumerate(train_loader):
            # Move the input and target tensors to the device

            data_t = data[0].to(device)
            data_target_position = data[1].to(device)
            data_source_activity_target = data[2].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data_t.float().reshape(-1,1))

            # Compute the loss
            loss = criterion(outputs, (data_target_position, data_source_activity_target))
            
            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Append the loss to the list of train losses
            train_loss.append(loss.item())

            train_loss_in_epoch.append(loss.item())

        train_loss_epochs.append(np.mean(train_loss_in_epoch))

        # Evaluate the model on the val dataset
        with torch.no_grad():
            val_loss_epoch = 0.0
            n_val_samples = 0
            for _, data in enumerate(val_loader):
                # Move the input and target tensors to the device
                data_t = data[0].to(device)
                data_target_position = data[1].to(device)
                data_source_activity_target = data[2].to(device)

                # Forward pass
                outputs = model(data_t.float().reshape(-1,1))

                # Compute the loss
                loss = criterion(outputs,(data_target_position, data_source_activity_target))

                # Accumulate the loss over the entire val set
                val_loss_epoch += loss.item() * data[0].shape[0]
                n_val_samples += batch_size
            
            # Divide the total loss by the number of samples to get the average val loss
            val_loss_epoch /= n_val_samples

            # Append the val loss to the list of val losses
            val_loss.append(val_loss_epoch)

        print('Epoch {} val Loss = {}'.format(epoch, val_loss_epoch))

        # Check for improvement in val_loss
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping due to no improvement in validation loss for {} epochs.".format(patience))
                break

        save_path =  os.path.join(checkpoint_directory,f"model_{name}_seed_{seed}_checkpoint_epoch{epoch}_valloss_{val_loss_epoch:.5f}.pt")

        if save_path is not None and save_freq is not None and (epoch + 1) % save_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, save_path)

        keep_best_checkpoint(checkpoint_directory)

    if ensemble_mode is True:
        best_checkpoint_path = glob.glob(os.path.join(checkpoint_directory, '*.pt'))[0]
        shutil.move(best_checkpoint_path, 'checkpoints/checkpoints_ensembles_saved')

    if plot_losses is True : 

        fig, ax = plt.subplots()
        ax.plot(train_loss_epochs, label='Training Loss')
        ax.plot(val_loss, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and validation losses over iterations')
        ax.legend()
        display(fig)
        plt.close()

def main(config):

    device = config['training']['device'] # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config['training']['seed']) 

    #Instantiate the chosen model and the corresponding loss
    if config['model']['name'] == "smcl":
        model = smcl(num_hypothesis=config['model']['num_hypothesis']).to(device)
        criterion = mhloss(
        mode=config['training']['wta_mode'],
    )
        
    elif config['model']['name'] == "singlesmcl":
        model = smcl(num_hypothesis=1).to(device)
        criterion = mhloss(
        mode=config['training']['wta_mode'],single_target_loss = True
    )

    elif config['model']['name'] == "rmcl":
        model = rmcl(num_hypothesis=config['model']['num_hypothesis']).to(device)
        criterion = rmcl_loss(
        number_unconfident=config['training']['number_unconfident'],
        mode=config['training']['wta_mode'],
        rejection_method=config['training']['rejection_method'],
        epsilon=config['training']['epsilon']
    )
        
    else:
        raise ValueError(f"Invalid model name '{config['model']['name']}' in the configuration.")

    # Define the optimizer
    optimizer = getattr(optim, config['training']['optimizer'])(model.parameters(), lr=config['training']['learning_rate'])

    # Create an instance of the ToyDataset class for the train dataset
    train_dataset = MultiSourcesToyDataset(n_samples=config['dataset']['n_samples_train'])
    train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True)

    # Create an instance of the ToyDataset class for the val dataset
    val_dataset = MultiSourcesToyDataset(n_samples=config['dataset']['n_samples_val'])
    val_loader = DataLoader(val_dataset, batch_size=config['dataset']['batch_size'], shuffle=False)

    train_model(
        name=config['model']['name'],
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
        device=device,
        save_freq=config['training']['checkpoint_frequency'],
        checkpoint_directory=config['training']['checkpoint_directory'],
        batch_size = config['dataset']['batch_size'],
        plot_losses = config['training']['plot_losses'],
        seed=config['training']['seed'],
        ensemble_mode=config['training']['ensemble_mode'],
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Argument parser config file choice')
    parser.add_argument('--override_config_path', type=str, default="config/override_config_rmcl.yaml", help='Path of the override config path')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for training')
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for computation (cuda or cpu)")
    args = parser.parse_args()

    # Load and merge the config files
    merged_config = load_and_merge_configs("config/config.yaml", args.override_config_path)

    # Set the random seed if provided
    if args.seed is not None:
        merged_config['training']['seed'] = args.seed

    merged_config['training']['device'] = torch.device(args.device)

    # Call the main function with the merged config
    main(merged_config)