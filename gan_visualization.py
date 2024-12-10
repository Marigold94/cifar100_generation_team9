import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torchvision
import seaborn as sns
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from tqdm import tqdm


class GANAnalyzer:
    def __init__(self, save_dir='./gan_analysis'):
        """
        Initialize GANAnalyzer with directory for saving plots
        
        Args:
            save_dir: Directory to save generated plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # CIFAR-100 superclass mapping
        self.superclass_mapping = {
            0: [4, 30, 55, 72, 95],  # aquatic mammals
            1: [1, 32, 67, 73, 91],  # fish
            2: [54, 62, 70, 82, 92],  # flowers
            3: [9, 10, 16, 28, 61],  # food containers
            4: [0, 51, 53, 57, 83],  # fruit and vegetables
            5: [22, 39, 40, 86, 87],  # household electrical devices
            6: [5, 20, 25, 84, 94],  # household furniture
            7: [6, 7, 14, 18, 24],  # insects
            8: [3, 42, 43, 88, 97],  # large carnivores
            9: [12, 17, 37, 68, 76],  # large man-made outdoor things
            10: [23, 33, 49, 60, 71],  # large natural outdoor scenes
            11: [15, 19, 21, 31, 38],  # large omnivores and herbivores
            12: [34, 63, 64, 66, 75],  # medium-sized mammals
            13: [26, 45, 77, 79, 99],  # non-insect invertebrates
            14: [2, 11, 35, 46, 98],  # people
            15: [27, 29, 44, 78, 93],  # reptiles
            16: [36, 50, 65, 74, 80],  # small mammals
            17: [47, 52, 56, 59, 96],  # trees
            18: [8, 13, 48, 58, 90],  # vehicles 1
            19: [41, 69, 81, 85, 89]   # vehicles 2
        }

    def save_samples(self, samples, filename):
        """Save generated sample images to a grid"""
        # Denormalize the images from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2.0
    
        # Create a grid of images
        grid = torchvision.utils.make_grid(
            samples,
            nrow=10,  # 10 images per row
            padding=2,
            normalize=False
        )
    
        # Convert to PIL image
        grid = grid.cpu().detach()
        grid = torch.clamp(grid, 0, 1)  # Ensure values are in [0, 1]
        grid = grid.permute(1, 2, 0).numpy()  # CHW -> HWC
    
        # Save the image
        plt.figure(figsize=(20, 20))
        plt.imshow(grid)
        plt.axis('off')
        plt.savefig(f'{self.save_dir}/{filename}', bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Saved sample images to {self.save_dir}/{filename}")

    def plot_losses(self, g_losses, d_losses, save=True):
        """Plot Generator and Discriminator losses over epochs"""
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label='Generator Loss')
        plt.plot(d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Progress')
        plt.legend()
        plt.grid(True)
        
        if save:
            plt.savefig(f'{self.save_dir}/loss_plot.png')
            plt.close()
        else:
            plt.show()

    def plot_tsne(self, real_features, fake_features, labels, save=True):
        """Generate t-SNE visualization of real and generated samples"""
        # Combine features
        combined_features = np.vstack([real_features, fake_features])
        
        # Create labels for real and fake
        combined_labels = np.concatenate([
            labels[:len(real_features)],
            labels[len(real_features):]])
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(combined_features)
        
        # Split back into real and fake
        real_tsne = features_tsne[:len(real_features)]
        fake_tsne = features_tsne[len(real_features):]
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter_real = plt.scatter(real_tsne[:, 0], real_tsne[:, 1], 
                                 c=labels[:len(real_features)], 
                                 alpha=0.5, label='Real')
        scatter_fake = plt.scatter(fake_tsne[:, 0], fake_tsne[:, 1], 
                                 c=labels[len(real_features):], 
                                 alpha=0.5, marker='x', label='Generated')
        
        plt.colorbar(scatter_real)
        plt.title('t-SNE Visualization of Real and Generated Samples')
        plt.legend()
        
        if save:
            plt.savefig(f'{self.save_dir}/tsne_plot.png')
            plt.close()
        else:
            plt.show()

    def plot_class_distribution(self, real_labels, fake_labels, save=True):
        """Plot distribution of classes in real and generated samples"""
        plt.figure(figsize=(15, 5))
        
        # Real samples
        plt.subplot(1, 2, 1)
        sns.histplot(real_labels, bins=20)
        plt.title('Real Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Generated samples
        plt.subplot(1, 2, 2)
        sns.histplot(fake_labels, bins=20)
        plt.title('Generated Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.save_dir}/class_distribution.png')
            plt.close()
        else:
            plt.show()

    def plot_feature_stats(self, real_features, fake_features, save=True):
        """Plot feature statistics comparison between real and generated samples"""
        real_mean = np.mean(real_features, axis=0)
        fake_mean = np.mean(fake_features, axis=0)
        real_std = np.std(real_features, axis=0)
        fake_std = np.std(fake_features, axis=0)
        
        # Plot means
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(real_mean, fake_mean, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
        plt.xlabel('Real Feature Means')
        plt.ylabel('Generated Feature Means')
        plt.title('Feature Means Comparison')
        
        # Plot standard deviations
        plt.subplot(1, 2, 2)
        plt.scatter(real_std, fake_std, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
        plt.xlabel('Real Feature Std')
        plt.ylabel('Generated Feature Std')
        plt.title('Feature Standard Deviations Comparison')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.save_dir}/feature_stats.png')
            plt.close()
        else:
            plt.show()

    def plot_inception_scores(self, inception_scores, save=True):
        """Plot inception scores over training"""
        plt.figure(figsize=(10, 5))
        means = [score[0] for score in inception_scores]
        stds = [score[1] for score in inception_scores]
        
        epochs = range(1, len(means) + 1)
        plt.errorbar(epochs, means, yerr=stds, fmt='-o')
        plt.xlabel('Epoch')
        plt.ylabel('Inception Score')
        plt.title('Inception Score Progress')
        plt.grid(True)
        
        if save:
            plt.savefig(f'{self.save_dir}/inception_scores.png')
            plt.close()
        else:
            plt.show()

    def plot_fid_scores(self, fid_scores, save=True):
        """Plot FID scores over training"""
        plt.figure(figsize=(10, 5))
        plt.plot(fid_scores, '-o')
        plt.xlabel('Epoch')
        plt.ylabel('FID Score')
        plt.title('FID Score Progress')
        plt.grid(True)
        
        if save:
            plt.savefig(f'{self.save_dir}/fid_scores.png')
            plt.close()
        else:
            plt.show()

    def generate_evaluation_report(self, metrics_dict, epoch):
        """Generate and save evaluation metrics report"""
        report = f"""
        GAN Evaluation Report - Epoch {epoch}
        ================================
        FID Score: {metrics_dict['fid']:.2f}
        Inception Score: {metrics_dict['inception_score_mean']:.2f} ± {metrics_dict['inception_score_std']:.2f}
        Intra-FID Score: {metrics_dict['intra_fid']:.2f}
        """
        
        with open(f'{self.save_dir}/evaluation_report_epoch_{epoch}.txt', 'w') as f:
            f.write(report)
        
        return report

def get_cifar100_loaders(batch_size=64, num_workers=2):
    """Get CIFAR-100 data loaders"""
    transform = transforms.Compose([
        transforms.Resize(64),  # Resize to match GAN output
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader