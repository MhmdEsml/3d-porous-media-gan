from dataclasses import dataclass
from typing import Tuple, Sequence

@dataclass
class Config:
    data_path = 'dataset'
    volume_dims = [1000, 1000, 1000]
    
    img_size = (64, 64, 64)
    latent_dim = 1
    g_features = [64, 128, 256]
    g_kernel_sizes = ((3, 3, 3), (3, 3, 3), (3, 3, 3))
    g_use_dilation = (False, False, False)
    g_use_attention = (False, False, False)
    d_features = [64, 128, 256]
    
    num_steps = 100_000
    batch_size_per_device = 1
    learning_rate_g = 2e-4
    learning_rate_d = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    
    lambda_adv = 1.0
    lambda_fm = 10.0
    lambda_porosity = 50.0
    
    log_every = 1_000
    save_every = 50_000
    save_dir = '/kaggle/working/checkpoints' # important note: the path should be absolute
    seed = 42
    
    viz_every = 5_000
    num_viz_samples = 8
    viz_dir = 'visualizations'
    
    resume = False
