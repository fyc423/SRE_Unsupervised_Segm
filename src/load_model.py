# load_model.py

import torch
import SRE_Conv.sre_resnet as sre_resnet
from SRE_Conv.sre_conv import SRE_Conv2d

def load_sre_model(weight_path=None, device='cuda'):
    # Initialize model with custom settings
    sre_model = getattr(sre_resnet, 'SRE_resnet18')(
        num_classes=9,
        kernel_shape='o',
        train_index_mat=True,
        inplanes=64,
        ri_conv_size=[9, 9, 5, 5],
        deepwise_ri=False,
        ri_index_mat_cin=False,
        ri_index_mat_cout=False,
        ri_split_ratio=0.5,
        large_conv=False,
        force_circular=True,
        gaussian_soft_kernel=False,
        train_gaussian_sigma=False
    )

    # Replace conv1 and maxpool
    sre_model.conv1 = SRI_Conv2d(
        3, 64, kernel_size=7, stride=1, padding=3, bias=False,
        kernel_shape='o', train_index_mat=False, inference_accelerate=True,
        force_circular=True, gaussian_soft_kernel=False, train_gaussian_sigma=False
    )
    sre_model.maxpool = torch.nn.Identity()

    # Load weights
    if weight_path is not None:
        print(f"🔄 Loading pretrained weights from: {weight_path}")
        sre_model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    else:
        print("⚠️ No weights provided. Initializing model with random weights.")

    # Truncate model before classifier
    sre_model_64 = torch.nn.Sequential(*(list(sre_model.children())[:-2]))

    # Freeze parameters
    for param in sre_model_64.parameters():
        param.requires_grad = False

    sre_model_64 = sre_model_64.to(device)
    sre_model_64.eval()

    return sre_model_64
