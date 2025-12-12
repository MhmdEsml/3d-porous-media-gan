import numpy as np
import jax.numpy as jnp
import os
from typing import Tuple, Dict, Any, Generator

def load_volume(raw_file_path: str, dims: Tuple[int, int, int]) -> np.ndarray:
    depth, height, width = dims
    data_type = np.uint8
    volume = np.fromfile(raw_file_path, dtype=data_type).reshape((depth, height, width))
    return volume

def create_train_generator(vol: np.ndarray, patch_size: Tuple[int, int, int], 
                          total_batch_size: int, compute_porosity: bool = True) -> Generator[Dict[str, Any], None, None]:
    d, h, w = vol.shape
    pd, ph, pw = patch_size
    z_max, y_max, x_max = d - pd, h - ph, w - pw
    
    while True:
        image_batch = []
        porosity_batch = []
        
        for _ in range(total_batch_size):
            z = np.random.randint(0, z_max + 1)
            y = np.random.randint(0, y_max + 1)
            x = np.random.randint(0, x_max + 1)
            
            patch = vol[z:z+pd, y:y+ph, x:x+pw].astype(np.float32)
            
            if np.random.random() > 0.5:
                patch = np.flip(patch, axis=0)
            if np.random.random() > 0.5:
                patch = np.flip(patch, axis=1)
            if np.random.random() > 0.5:
                patch = np.flip(patch, axis=2)
            
            if compute_porosity:
                porosity = np.mean(patch == 0).astype(np.float32)
                porosity_batch.append(porosity)
            
            image_batch.append(patch.copy())
        
        output = {
            'images': 2.0 * np.expand_dims(np.stack(image_batch), axis=-1) - 1.0,
        }
        
        if compute_porosity:
            output['porosity'] = np.array(porosity_batch)
        
        yield output
