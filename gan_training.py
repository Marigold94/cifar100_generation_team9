import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from improved_cifar_gan import BigGAN  # Changed from CIFAR100GAN to BigGAN
from gan_visualization import GANAnalyzer
from inception_model import load_inception_net

def get_cifar100_loaders(batch_size=128, num_workers=2):
    """Get CIFAR-100 data loaders"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
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
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, test_loader

def train_and_visualize():
    # Create necessary directories
    import os
    os.makedirs('./gan_results', exist_ok=True)
    os.makedirs('./gan_results/samples', exist_ok=True)
    os.makedirs('./gan_results/superclass', exist_ok=True)
    os.makedirs('./gan_results/metrics', exist_ok=True)
    os.makedirs('./gan_results/checkpoints', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize data loaders
    train_loader, test_loader = get_cifar100_loaders(batch_size=128)

    # Initialize GAN and analyzer
    gan = BigGAN(
        latent_dim=128,
        num_classes=100,
        lr_g=1e-4,
        lr_d=4e-4,
        n_critic=5
    )

    # Initialize analyzer and inception model
    analyzer = GANAnalyzer(save_dir='./gan_results')
    inception_net = load_inception_net(device)

    # Train the model
    num_epochs = 1
    current_epoch, metrics = gan.train(train_loader, num_epochs, device, analyzer)

    # Generate and save visualizations
    for epoch in range(num_epochs):
        if (epoch + 1) % 5 == 0:
            print("\nGenerating visualizations...")
            
            # Plot and save losses
            analyzer.plot_losses(gan.g_losses, gan.d_losses)
            
            # Generate sample images
            with torch.no_grad():
                samples = gan.generate_samples(100, device)
                torchvision.utils.save_image(
                    samples,
                    f'./gan_results/samples_epoch_{epoch+1}.png',
                    normalize=True,
                    nrow=10
                )

        # Save generated images and evaluate
        if (epoch + 1) % 20 == 0:
            print(f"\nGenerating images and evaluating for epoch {epoch + 1}...")
            
            # Generate images for each superclass
            for superclass in range(20):
                fine_classes = analyzer.superclass_mapping[superclass]
                all_samples = []
                
                with torch.no_grad():
                    for fine_class in fine_classes:
                        z = torch.randn(5, gan.latent_dim, device=device)
                        labels = torch.full((5,), fine_class, dtype=torch.long, device=device)
                        samples = gan.G(z, labels)
                        all_samples.append(samples)
                
                superclass_samples = torch.cat(all_samples, 0)
                torchvision.utils.save_image(
                    superclass_samples,
                    f'./gan_results/epoch_{epoch+1}_superclass_{superclass}.png',
                    normalize=True,
                    nrow=5
                )
            
            # Save random samples
            samples = gan.generate_samples(100, device)
            torchvision.utils.save_image(
                samples,
                f'./gan_results/epoch_{epoch+1}_random_samples.png',
                normalize=True,
                nrow=10
            )

            # Evaluate model
            metrics = gan.evaluate(train_loader, device, inception_net)
            report = analyzer.generate_evaluation_report(metrics, epoch + 1)
            print(f"\nEvaluation Report for epoch {epoch + 1}:")
            print(report)

    # Final evaluation
    print("\nPerforming final evaluation...")
    final_metrics = gan.evaluate(train_loader, device, inception_net)
    final_report = analyzer.generate_evaluation_report(final_metrics, 'final')
    print("\nFinal Evaluation Report:")
    print(final_report)

    # Save final model
    gan.save_checkpoint('gan_results/final_model.pt')

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    train_and_visualize()