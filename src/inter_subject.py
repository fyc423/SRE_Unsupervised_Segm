import os
import pandas as pd
import monai 
from monai.data.utils import list_data_collate,pad_list_data_collate
from monai.transforms import Resize
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.cluster import KMeans
from segmentation_plot import segmplot

from opt import get_args
from utils import channel_first, resizer, onehot, create_binary_mask, cluster_training, predict
from load_model import load_sre_model


if __name__ == '__main__':
    args = get_args()

    # Load training data
    df_train = pd.read_csv(args.csv_path)
    df_train_dict = df_train.to_dict(orient='records')

    # Load test data
    if args.test_data_path is not None:
        df_test = pd.read_csv(args.test_data_path)
        df_test_dict = df_test.to_dict(orient='records')
    else:
        df_test_dict = df_train[26:51].to_dict(orient='records')

    # Load model
    sre_model = load_sre_model(weight_path=args.weight_path, device='cuda')

    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)

    #----------------------------------------------------------------------#
    # Feature extraction from training data
    embedding_sri_list = []
    for idx in range(len(df_train_dict)):
        print(f'Embedding training subject {idx}...')
        train_transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=['processed'], reader='PydicomReader'),
            monai.transforms.EnsureChannelFirstd(keys=['processed']),
            monai.transforms.Resized(keys=['processed'], spatial_size=(512, 512)),
            monai.transforms.ToTensord(keys=['processed'])
        ])

        df_set = monai.data.Dataset(data=df_train_dict[idx:idx+1], transform=train_transforms)
        df_loader = DataLoader(df_set, num_workers=4, collate_fn=pad_list_data_collate)

        embedding_sri_subject_list = []
        img_subject_list = []

        for d in tqdm(df_loader):
            I = d['processed'].to('cuda')
            f = sre_model(I).detach().cpu().numpy()
            I = I.detach().cpu().numpy()
            embedding_sri_subject_list.append(f)
            img_subject_list.append(I)

        process_feat = np.vstack(embedding_sri_subject_list).squeeze()
        process_img = np.vstack(img_subject_list).squeeze()

        M = create_binary_mask(img=process_img, k=64, threshold=20, resizer=resizer)
        feat = np.expand_dims(resizer(process_feat), axis=0)

        embedding_sri_list.append(feat)

    total_train_list = np.concatenate(embedding_sri_list, axis=0)
    print('Total training features:', total_train_list.shape)
    

    embed_model, _, _ = cluster_training(
        feature_baseline=total_train_list,
        mask=M,
        n_clusters=args.n_clusters,
        n_samples=500
    )
    
    torch.cuda.empty_cache()

    #----------------------------------------------------------------------#
    # Start testing 
    for idx in range(len(df_test_dict)):
        subject = df_test_dict[idx:idx+1]

        for deg in range(0, 360, 30):
            print(f'Processing test subject {idx}, degree {deg}...')

            test_transforms = monai.transforms.Compose([
                monai.transforms.LoadImaged(keys=['processed'], reader='PydicomReader'),
                monai.transforms.EnsureChannelFirstd(keys=['processed']),
                monai.transforms.Resized(keys=['processed'], spatial_size=(512, 512)),
                monai.transforms.Rotated(keys=['processed'], angle=deg * np.pi / 180, keep_size=True, padding_mode='border'),
                monai.transforms.ToTensord(keys=['processed'])
            ])

            df_set = monai.data.Dataset(data=subject, transform=test_transforms)
            df_loader = DataLoader(df_set, num_workers=4, collate_fn=pad_list_data_collate)

            embedding_sri_list = []
            img_list = []

            for d in tqdm(df_loader):
                I = d['processed'].to('cuda')
                f = sre_model(I).detach().cpu().numpy()
                I = I.detach().cpu().numpy()
                embedding_sri_list.append(f)
                img_list.append(I)

                slide_id = str(d['slide'][0])
                core_id = str(d['core'][0])

            process_feat = np.vstack(embedding_sri_list).squeeze()
            process_img = np.vstack(img_list).squeeze()

            M = create_binary_mask(img=process_img, k=64, threshold=20, resizer=resizer)
            feat = np.expand_dims(resizer(process_feat), axis=0)
            resize_img = resizer(process_img).astype(int)

            folder_name = f"slide{slide_id}_core{core_id}_segmented"
            folder_path = os.path.join(args.output_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            save_path = os.path.join(folder_path, f"unsupervised_segment_deg_{deg}.pdf")
            _ = predict(embed_model=embed_model, input_feature=feat, input_img=resize_img,
                        mask=M, angle=deg, save_path=save_path)

        torch.cuda.empty_cache()
        print(f"Finished subject {idx} | Slide {slide_id} | Core {core_id} | Rotation Degree {deg}")

            
            

    
            



