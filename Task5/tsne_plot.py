import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

def get_image_embeddings(image_dir, model, transform, device, max_images=1000):
    embeddings = []
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))][:max_images]
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc=f"Processing {os.path.basename(image_dir)}"):
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                feat = model(tensor).squeeze().cpu().numpy()
                embeddings.append(feat)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
    return np.array(embeddings)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weights = ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    resnet.fc = torch.nn.Identity() 
    resnet = resnet.to(device).eval()

    transform = weights.transforms()

    real_dir = "/ghome/group05/datasets/VizWiz/train"
    synth_dir = "/ghome/group05/datasets/synthetic_vizwiz_blurry/images"
    deart_dir = "/ghome/group05/datasets/deART/images"

    print("Extracting features...")
    real_features = get_image_embeddings(real_dir, resnet, transform, device, max_images=3000)
    synth_features = get_image_embeddings(synth_dir, resnet, transform, device, max_images=3000)
    deart_features = get_image_embeddings(deart_dir, resnet, transform, device, max_images=3000)

    X = np.vstack((real_features, synth_features, deart_features))
    labels = np.array([0] * len(real_features) + [1] * len(synth_features) + [2] * len(deart_features))

    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)

    # Increase base font size for axis ticks and labels
    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(12, 9))
    # Increased marker size (s) and adjusted alpha for presentation clarity
    plt.scatter(X_2d[labels == 0, 0], X_2d[labels == 0, 1], c='red', alpha=0.6, label='Real VizWiz', edgecolors='none', s=40)
    plt.scatter(X_2d[labels == 1, 0], X_2d[labels == 1, 1], c='orange', alpha=0.7, label='Synthetic SD3', edgecolors='none', s=40)
    plt.scatter(X_2d[labels == 2, 0], X_2d[labels == 2, 1], c='blue', alpha=0.5, label='deART dataset', edgecolors='none', s=40)
    
    plt.title('t-SNE Visualization of Image Feature Distributions', fontsize=20, pad=20)
    plt.legend(fontsize=14, markerscale=1.5, loc='upper right')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('distribution_comparison_tsne.png', dpi=300, bbox_inches='tight')
    print("✅ Plot saved to distribution_comparison_tsne.png")

if __name__ == "__main__":
    main()