import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3
import numpy as np

class WrapInception(nn.Module):
    def __init__(self, net):
        super(WrapInception,self).__init__()
        self.net = net
        self.mean = nn.Parameter(
            torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
            requires_grad=False)
        self.std = nn.Parameter(
            torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
            requires_grad=False)

    def forward(self, x):
        # Normalize x
        x = (x + 1.) / 2.0
        x = (x - self.mean) / self.std
        
        # Upsample if necessary
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
            
        # 299 x 299 x 3
        x = self.net.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.net.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.net.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.net.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.net.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.net.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.net.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.net.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.net.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.net.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.net.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.net.Mixed_7c(x)
        # 8 x 8 x 2048
        
        # Global average pooling
        pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        # Compute logits
        logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
        
        return pool, logits

def load_inception_net(device=None):
    """Load and wrap inception v3 model"""
    # 최신 weights enum 사용
    from torchvision.models.inception import Inception_V3_Weights
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception_model = WrapInception(inception_model.eval())
    if device is not None:
        inception_model = inception_model.to(device)
    return inception_model

def accumulate_inception_activations(data_loader, model, device, max_samples=50000):
    """Accumulate inception activations for real/generated images"""
    pool = []
    logits = []
    labels = []
    
    for i, (images, batch_labels) in enumerate(data_loader):
        if len(pool) * images.size(0) >= max_samples:
            break
            
        images = images.to(device)
        with torch.no_grad():
            pool_val, logits_val = model(images)
            pool.append(pool_val.cpu().numpy())
            logits.append(F.softmax(logits_val, dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
            
    return np.concatenate(pool), np.concatenate(logits), np.concatenate(labels)

def calculate_inception_score(pred, num_splits=10):
    scores = []
    for index in range(num_splits):
        pred_chunk = pred[index * (pred.shape[0] // num_splits): 
                         (index + 1) * (pred.shape[0] // num_splits), :]
        kl_inception = pred_chunk * \
            (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
        kl_inception = np.mean(np.sum(kl_inception, 1))
        scores.append(np.exp(kl_inception))
        
    return np.mean(scores), np.std(scores)