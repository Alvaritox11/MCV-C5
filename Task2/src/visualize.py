import numpy as np
import matplotlib.pyplot as plt
import wandb
import io
import cv2
from PIL import Image

def show_mask(mask, ax, random_color=False):
    """Applies a mask overlay to a matplotlib axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Default blue-ish color with alpha=0.6
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=200):
    """Plots positive (green) and negative (red) prompt points."""
    coords = np.array(coords)
    labels = np.array(labels)
    
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                   s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
                   s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """Plots a bounding box."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def plot_prediction(image, masks, points=None, labels=None, boxes=None, title="", figsize=(10, 10)):
    """
    High-level function ideal for Jupyter Notebooks.
    Call this and it will render the matplotlib plot inline.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    
    # Plot all masks
    for mask in masks:
        show_mask(mask, ax, random_color=True)
        
    # Plot points if provided
    if points is not None and labels is not None:
        for pt_group, lbl_group in zip(points, labels):
            show_points(pt_group, lbl_group, ax)
            
    # Plot boxes if provided
    if boxes is not None:
        for box in boxes:
            show_box(box, ax)
            
    ax.set_title(title)
    ax.axis('off')
    plt.show()

def create_wandb_image(image, masks, points=None, labels=None, boxes=None, title=""):
    """
    Generates a plot and converts it to a wandb.Image object without displaying it.
    This is what main.py will use to log to Weights & Biases.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    
    for mask in masks:
        show_mask(mask, ax, random_color=True)
        
    if points is not None and labels is not None:
        for pt_group, lbl_group in zip(points, labels):
            show_points(pt_group, lbl_group, ax)
            
    if boxes is not None:
        for box in boxes:
            show_box(box, ax)
            
    ax.set_title(title)
    ax.axis('off')
    
    # Save figure to a buffer so we don't open hundreds of windows during evaluation
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf)
    return wandb.Image(img, caption=title)


def plot_presentation_quality(image, masks, boxes, labels, scores, title="", figsize=(12, 8)):
    """
    Creates a highly polished visualization using OpenCV.
    Draws boxes, writes labels with confidence scores, and alpha-blends masks.
    """
    # Convert PIL Image to OpenCV format (BGR)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Create an overlay for the transparent masks
    overlay = img_cv.copy()
    
    # Generate random distinct colors for each detection
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(masks))]
    
    for i in range(len(masks)):
        mask = masks[i]
        box = boxes[i]
        label = labels[i]
        score = scores[i]
        color = colors[i]
        
        # 1. Draw mask on the overlay
        overlay[mask > 0] = color
        
        # 2. Draw bounding box on the main image
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        
        # 3. Draw text banner (Label + Score)
        text = f"{label} ({score:.2f})"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(img_cv, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), color, -1)
        # Draw text in white
        cv2.putText(img_cv, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 4. Blend the mask overlay with the original image
    alpha = 0.4 # Transparency factor
    img_cv = cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0)
    
    # Convert back to RGB for matplotlib plotting
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    return img_rgb # Return the array in case you want to save it directly