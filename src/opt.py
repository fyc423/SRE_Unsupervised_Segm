import argparse

def get_opts():
    parser = argparse.ArgumentParser(description="Options for Unsupervised Segmentation Using SRE-Conv model")

    # Input data path
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file containing image paths (in inter-sbuject analysis the data is used for cluster training)")
    
    # Optional test data path for inter-subject analysis
    parser.add_argument("--test_data_path", type=str, default=None,
    help="Optional path to test data CSV (used for inter-subject analysis)")
    
    # Segmented image output path
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory to save output files (e.g. features, predictions, etc.)")

    # Number of clusters
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters for KMeans or similar')

    # Number of samples used for cluster model training
    parser.add_argument("--n_samples", type=int, default=2000,
    help="Number of samples from clustering baseline used in cluster_training"


    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    
    # Pretrained model weights
    parser.add_argument("--weight_path", type=str, required=True,
                    help="Path to pretrained model weights")



    return parser.parse_args()
