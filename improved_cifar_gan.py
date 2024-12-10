import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.cuda.amp import autocast, GradScaler

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # gamma
        self.embed.weight.data[:, num_features:].zero_()  # beta

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return gamma * out + beta

class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.bypass = nn.Sequential()
        if in_channels != out_channels:
            self.bypass = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        
        self.bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)

    def forward(self, x, y):
        h = self.bn1(x, y)
        h = F.relu(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.conv1(h)
        h = self.bn2(h, y)
        h = F.relu(h)
        h = self.conv2(h)
        
        x = F.interpolate(x, scale_factor=2)
        x = self.bypass(x)
        
        return h + x

class Generator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=100):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Initial dense layer
        self.fc = spectral_norm(nn.Linear(latent_dim, 4 * 4 * 256))
        
        # Residual blocks with upsampling
        self.res1 = ResBlockGenerator(256, 256, num_classes)  # 4x4 -> 8x8
        self.res2 = ResBlockGenerator(256, 128, num_classes)  # 8x8 -> 16x16
        self.res3 = ResBlockGenerator(128, 64, num_classes)   # 16x16 -> 32x32
        
        # Final conv layer
        self.final = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 3, 3, 1, 1)),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, z, y):
        h = self.fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        h = self.res1(h, y)
        h = self.res2(h, y)
        h = self.res3(h, y)
        return self.final(h)

class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.bypass = nn.Sequential()
        if in_channels != out_channels:
            self.bypass = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        
        self.downsample = downsample

    def forward(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
        x = self.bypass(x)
        return h + x

class Discriminator(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        
        # Initial conv layer
        self.conv1 = spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        
        # Residual blocks with downsampling
        self.res1 = ResBlockDiscriminator(64, 128)    # 32x32 -> 16x16
        self.res2 = ResBlockDiscriminator(128, 256)   # 16x16 -> 8x8
        self.res3 = ResBlockDiscriminator(256, 512)   # 8x8 -> 4x4
        self.res4 = ResBlockDiscriminator(512, 1024, downsample=False)  # 4x4
        
        # Output layers
        self.fc = spectral_norm(nn.Linear(1024, 1))
        self.embed = spectral_norm(nn.Embedding(num_classes, 1024))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
            nn.init.orthogonal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        h = self.conv1(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = F.relu(h)
        h = torch.sum(h, dim=(2, 3))  # Global sum pooling
        
        output = self.fc(h)
        embed = self.embed(y)
        # Projection discriminator
        output += torch.sum(embed * h, dim=1, keepdim=True)
        
        return output

class BigGAN:
    def __init__(self, latent_dim=128, num_classes=100, lr_g=1e-4, lr_d=4e-4, n_critic=5):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.n_critic = n_critic
        
        # Initialize models
        self.G = Generator(latent_dim, num_classes)
        self.D = Discriminator(num_classes)
        
        # Optimizers
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.0, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.0, 0.999))
        
        # Mixed precision training
        self.scaler_G = GradScaler()
        self.scaler_D = GradScaler()
        
        # Training history
        self.g_losses = []
        self.d_losses = []

    def train_discriminator(self, real_imgs, labels, device):
        batch_size = real_imgs.size(0)
        
        # Sample random latent vectors
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        with autocast():
            # Generate fake images
            fake_imgs = self.G(z, labels)
            
            # Real images
            real_validity = self.D(real_imgs, labels)
            d_real_loss = F.relu(1.0 - real_validity).mean()
            
            # Fake images
            fake_validity = self.D(fake_imgs.detach(), labels)
            d_fake_loss = F.relu(1.0 + fake_validity).mean()
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
        
        self.opt_D.zero_grad()
        self.scaler_D.scale(d_loss).backward()
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()
        
        return d_loss.item()

    def train_generator(self, real_imgs, labels, device):
        batch_size = real_imgs.size(0)
        
        # Sample random latent vectors
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        with autocast():
            # Generate fake images
            fake_imgs = self.G(z, labels)
            
            # Calculate generator loss
            fake_validity = self.D(fake_imgs, labels)
            g_loss = -fake_validity.mean()
        
        self.opt_G.zero_grad()
        self.scaler_G.scale(g_loss).backward()
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()
        
        return g_loss.item()
    
    def train(self, dataloader, epochs, device, analyzer=None):
        print("Moving models to", device)
        self.G.to(device)
        self.D.to(device)
        
        from tqdm import tqdm
        total_batches = len(dataloader)
        
        for epoch in range(epochs):
            progress_bar = tqdm(enumerate(dataloader), total=total_batches)
            progress_bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            
            for i, (real_imgs, labels) in progress_bar:
                real_imgs = real_imgs.to(device)
                labels = labels.to(device)
                
                # Train discriminator
                d_loss = self.train_discriminator(real_imgs, labels, device)
                
                # Train generator every n_critic iterations
                if i % self.n_critic == 0:
                    g_loss = self.train_generator(real_imgs, labels, device)
                    progress_bar.set_postfix({
                        'g_loss': f"{g_loss:.4f}",
                        'd_loss': f"{d_loss:.4f}"
                    })
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"\nSaving checkpoint at epoch {epoch+1}")
                self.save_checkpoint(f"./gan_results/checkpoints/checkpoint_epoch_{epoch+1}.pt")
            
            # Save samples with analyzer if provided
            if analyzer is not None and (epoch + 1) % 5 == 0:
                print(f"\nGenerating samples at epoch {epoch+1}")
                with torch.no_grad():
                    samples = self.generate_samples(25, device)
                    analyzer.save_samples(samples, f"samples_epoch_{epoch+1}.png")

        print("\n=== Training completed. Running final evaluation... ===")
        print("This might take a while...")
        metrics = self.evaluate(dataloader, device)
        
        print("\n=== Final Evaluation Results ===")
        print(f"FID Score: {metrics['fid']:.2f}")
        print(f"Inception Score: {metrics['inception_score_mean']:.2f} ± {metrics['inception_score_std']:.2f}")
        print(f"Intra-FID Score: {metrics['intra_fid']:.2f}")
        
        # Save final model with metrics
        final_path = './gan_results/checkpoints/final_model.pt'
        self.save_checkpoint(final_path, metrics)
        print(f"Saved final model and metrics to {final_path}")
        
        return metrics

    def evaluate(self, real_loader, device, inception_net=None):
        """Evaluate the model using FID and Inception Score"""
        if inception_net is None:
            from inception_model import load_inception_net
            inception_net = load_inception_net(device)
        
        # Generate samples for evaluation
        num_samples = len(real_loader.dataset)
        generated_images = []
        generated_labels = []
        
        self.G.eval()
        with torch.no_grad():
            for _ in range(0, num_samples, 100):
                current_batch_size = min(100, num_samples - len(generated_images))
                z = torch.randn(current_batch_size, self.latent_dim, device=device)
                labels = torch.randint(0, self.num_classes, (current_batch_size,), device=device)
                samples = self.G(z, labels)
                generated_images.append(samples)
                generated_labels.append(labels)
        
        generated_images = torch.cat(generated_images, dim=0)
        generated_labels = torch.cat(generated_labels, dim=0)
        
        # Calculate metrics
        from evaluation_metrics import evaluate_gan
        metrics = evaluate_gan(real_loader, generated_images, generated_labels, inception_net)
        
        self.G.train()
        return metrics

    def generate_samples(self, n_samples, device):
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            labels = torch.randint(0, self.num_classes, (n_samples,), device=device)
            samples = self.G(z, labels)
        self.G.train()
        return samples

    def save_checkpoint(self, filename):
        torch.save({
            'g_state_dict': self.G.state_dict(),
            'd_state_dict': self.D.state_dict(),
            'opt_g_state_dict': self.opt_G.state_dict(),
            'opt_d_state_dict': self.opt_D.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
        }, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.G.load_state_dict(checkpoint['g_state_dict'])
        self.D.load_state_dict(checkpoint['d_state_dict'])
        self.opt_G.load_state_dict(checkpoint['opt_g_state_dict'])
        self.opt_D.load_state_dict(checkpoint['opt_d_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']