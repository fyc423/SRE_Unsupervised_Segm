import os
import pandas as pd
import monai 
from monai.data.utils import list_data_collate,pad_list_data_collate
from monai.transforms import EnsureChannelFirst, Resize, AsDiscrete
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.cluster import KMeans
from segmentation_plot import segmplot

from monai.transforms import EnsureChannelFirst, Resize, AsDiscrete


# Ensures channel-first layout for input images
channel_first = EnsureChannelFirst(channel_dim=-1)

# Resizes image to 128x128
resizer = Resize(spatial_size=(128, 128))

# Converts labels to one-hot with 4 classes
onehot = AsDiscrete(to_onehot=4)

def create_binary_mask(img, k=64, threshold=40, resizer=None):
    """
    Create a binary mask from an input image.
    
    Parameters:
    - img: np.ndarray, input image
    - k: int, the size to which the image should be resized (default 64)
    - threshold: float, intensity threshold for masking (default 40)
    - resizer: callable, function to resize the image back to original size
    
    Returns:
    - mask: np.ndarray, binary mask of the same size as the input image
    """
    # Resize the image
    resize_transform = monai.transforms.Resize(spatial_size=(k, k))
    J = resize_transform(img)
    
    # Invert the image intensities
    J_mag = np.sqrt(np.sum(J*J, axis=0, keepdims=True))
    J_mask = np.zeros((1, k, k))
    J_mask[J_mag > threshold] = 1

    # Resize image of original size 
    if resizer is not None:
        M = np.array(resizer(J_mask, mode="nearest"))
    else:
        M = np.array(J_mask)

    
    return M


def cluster_training(feature_baseline, mask, n_clusters=3, n_samples=500):
    """
    Training Kmeans model and cluster the feature map using K-means clustering based on the provided mask.
    
    Parameters:
    - feature_baseline: np.ndarray, the baseline input feature map 
    - mask: np.ndarray, binary mask to select relevant parts of the feature map
    - n_clusters: int, the number of clusters for K-means (default 3)
    - n_samples: int, the number of subsamples for K-means (default 500)
    
    Returns:
    - embed_model: KMeans, the fitted KMeans model
    - features: np.ndarray, predicted cluster labels for the entire mask
    - img_feat: np.ndarray, predicted cluster label in original position of image 
    """
    # Get the indexes where the mask is true
    idx_all = np.where(mask > 0)
    print("Total number of samples in mask =", idx_all[0].shape[0])

    # Subsample the indexes if necessary
    if idx_all[0].shape[0] > n_samples:
        idx_subsample = np.random.choice(idx_all[0].shape[0], n_samples, replace=False)
    else:
        idx_subsample = np.arange(idx_all[0].shape[0])
    print("Subsample shape:", idx_subsample.shape)
    
    # Extract the training samples from the feature map using the subsampled indexes
    X_train = feature_baseline[0,:, idx_all[1][idx_subsample], idx_all[2][idx_subsample]]
    print("Trainin Sample.shape:", X_train.shape)
    
    # Perform K-means clustering
    embed_model = KMeans(n_clusters=n_clusters, random_state = seed)
    X_train = X_train / np.linalg.norm(X_train, axis=1)[:, None]
    embed_model.fit(X_train)
    
    # Predict on all valid voxels
    X_test = feature_baseline[0,:, idx_all[1], idx_all[2]]
    X_test = X_test / np.linalg.norm(X_test, axis=1)[:, None]
    features = embed_model.predict(X_test)
    print(".shape:", features.shape)
    
    img_feat = np.zeros(mask.shape)
    # Add a fake 1 here to make it different from background (not necessary in real life)
    img_feat[0,idx_all[1],idx_all[2]] = features+1
    
    return embed_model, features, img_feat


def predict(embed_model, input_feature, input_img, input_mask, angle, save_path=None):

    """
    Using previous trained Kmeans model to predict cluster label of rotated image
    
    Parameters:
    - embed_model: KMeans, the fitted KMeans model
    - input_feature: np.ndarray, the input rotate image feature map 
    - input_img: np.ndarray, the input rotate image for plotting purpose
    - input_mask: np.ndarray, binary mask to select relevant parts of the feature map

    
    Returns:
    - fig: plot the following
        - rotated image
        - overlay of cluster label outlined area on rotated image
        - overlay of cluster label segmented area on rotated image
        - totated segmentation back to original orientation 
    """
    
    unrotater = monai.transforms.Rotate(angle=-angle*np.pi/180, padding_mode="zeros", mode="nearest")

    image_rot = input_image.astype(int)
    mask_rot = input_mask.astype(int)

    # Classify using the embedding model
    # Get the image indexes where the Mask is true
    idx_all = np.where(mask_rot>0)

    # Predict on all valid voxels
    X_test = input_feature[0,:,idx_all[1],idx_all[2]]
    X_test = X_test / np.linalg.norm(X_test, axis=1)[:, None]
    Y_pred = embed_model.predict(X_test)
    print('Predicted features',Y_pred.shape)
    
    # print('feature_baseline',baseline_feature.shape)

    # Put the cluster labels back into the image structure
    M_pred = np.zeros(mask_rot.shape)
    M_pred[0,idx_all[1],idx_all[2]] = Y_pred+1

    # Plot the images 
    fig = plt.figure(figsize=plt.figaspect(0.25))
    # Plot rotated image
    plt.subplot(1,4,1)
    segmplot.plot_segmentation(
        image=image_rot,
        segm=[onehot(mask_rot)],
        smooth_sigma=1.0,
        threshold=0.7,
        linewidth=2.0,
    )
    plt.title("Image ({})".format(angle))
    #Plot overlay of cluster label outlined area on rotated image
    plt.subplot(1,4,2)
    segmplot.plot_segmentation(
        image=image_rot,
        segm=[onehot(M_pred)],
        smooth_sigma=1.0,
        threshold=0.7,
        linewidth=2.0,
    )
    plt.title("Segm ({})".format(angle))
    #Plot overlay of cluster label segmented area on rotated image
    plt.subplot(1,4,3)
    im = segmplot.plot_segmentation(
        image=M_pred,
        segm=[onehot(M_pred)],
        smooth_sigma=1.0,
        threshold=0.7,
        linewidth=2.0,
        image_vmin= 0.0,
        image_vmax= 3.0,
    )
    # plt.colorbar(im, ax=plt.gca())
    plt.title("Segm ({})".format(angle))
    #Plot segmented area and rotate back to original orientation 
    plt.subplot(1,4,4)
    im2 = segmplot.plot_segmentation(
        image=unrotater(M_pred),
        segm=[onehot(unrotater(M_pred))],
        smooth_sigma=1.0,
        threshold=0.7,
        linewidth=2.0,
        image_vmin= 0.0,
        image_vmax= 3.0,
    )
    # plt.colorbar(im2, ax=plt.gca())
    plt.title("Unrotated Segm ({})".format(angle))
    # plt.show()

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close('all')

    return fig