import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm

def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def calculate_inception_score(pred, num_splits=10):
    scores = []
    for index in range(num_splits):
        pred_chunk = pred[index * (pred.shape[0] // num_splits): 
                         (index + 1) * (pred.shape[0] // num_splits), :]
        kl_inception = pred_chunk * \
            (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
        kl_inception = np.mean(np.sum(kl_inception, 1))
        scores.append(np.exp(kl_inception))
    return float(np.mean(scores)), float(np.std(scores))

def calculate_intra_fid(real_features, fake_features, real_labels, fake_labels):
    intra_fids = []
    unique_labels = np.unique(real_labels)
    for label in unique_labels:
        real_idx = real_labels == label
        fake_idx = fake_labels == label
        if np.sum(real_idx) < 2 or np.sum(fake_idx) < 2:
            continue
        intra_fid = calculate_fid(real_features[real_idx], fake_features[fake_idx])
        intra_fids.append(intra_fid)
    return float(np.mean(intra_fids))

def evaluate_gan(real_loader, fake_images, fake_labels, inception_net):
    print("\nExtracting features from real images...")
    real_features = []
    real_probs = []
    real_labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(real_loader, desc="Processing real images"):
            images = images.cuda()
            features, logits = inception_net(images)
            probs = F.softmax(logits, dim=1)
            real_features.append(features.cpu().numpy())
            real_probs.append(probs.cpu().numpy())
            real_labels_list.append(labels.numpy())
    
    real_features = np.concatenate(real_features, axis=0)
    real_labels = np.concatenate(real_labels_list, axis=0)
    
    print("Extracting features from generated images...")
    with torch.no_grad():
        fake_features = []
        fake_probs = []
        
        for i in tqdm(range(0, len(fake_images), 100), desc="Processing generated images"):
            batch = fake_images[i:i+100]
            features, logits = inception_net(batch)
            probs = F.softmax(logits, dim=1)
            fake_features.append(features.cpu().numpy())
            fake_probs.append(probs.cpu().numpy())
    
    fake_features = np.concatenate(fake_features, axis=0)
    fake_probs = np.concatenate(fake_probs, axis=0)
    
    print("\nComputing metrics...")
    fid = calculate_fid(real_features, fake_features)
    is_mean, is_std = calculate_inception_score(fake_probs)
    intra_fid = calculate_intra_fid(real_features, fake_features, 
                                   real_labels, fake_labels.cpu().numpy())
    
    return {
        'fid': fid,
        'inception_score_mean': is_mean,
        'inception_score_std': is_std,
        'intra_fid': intra_fid
    }