import os
import urllib.parse
import urllib.request
from typing import List

DATASETS = {
    "Berea": "Berea/Berea_2d25um_binary.raw/Berea_2d25um_binary.raw",
    "BanderaBrown": "Bandera Brown/BanderaBrown_2d25um_binary.raw/BanderaBrown_2d25um_binary.raw",
    "BanderaGray": "Bandera Gray/BanderaGray_2d25um_binary.raw/BanderaGray_2d25um_binary.raw",
    "Bentheimer": "Bentheimer/Bentheimer_2d25um_binary.raw/Bentheimer_2d25um_binary.raw",
    "BSG": "Berea Sister Gray/BSG_2d25um_binary.raw/BSG_2d25um_binary.raw",
    "BUG": "Berea Upper Gray/BUG_2d25um_binary.raw/BUG_2d25um_binary.raw",
    "BuffBerea": "Buff Berea/BB_2d25um_binary.raw/BB_2d25um_binary.raw",
    "CastleGate": "CastleGate/CastleGate_2d25um_binary.raw/CastleGate_2d25um_binary.raw",
    "Kirby": "Kirby/Kirby_2d25um_binary.raw/Kirby_2d25um_binary.raw",
    "Leopard": "Leopard/Leopard_2d25um_binary.raw/Leopard_2d25um_binary.raw",
    "Parker": "Parker/Parker_2d25um_binary.raw/Parker_2d25um_binary.raw",
}

BASE_URL = "https://web.corral.tacc.utexas.edu/digitalporousmedia/DRP-317"

def download_dataset(selected: List[str], data_path: str = "dataset") -> None:
    os.makedirs(data_path, exist_ok=True)
    
    for name in selected:
        if name not in DATASETS:
            continue
        
        rel = DATASETS[name]
        encoded = "/".join(urllib.parse.quote(p) for p in rel.split("/"))
        url = f"{BASE_URL}/{encoded}"
        out_path = os.path.join(data_path, f"{name}.raw")
        
        try:
            print(f"Downloading: {url}")
            urllib.request.urlretrieve(url, out_path)
            print(f"Saved to: {out_path}")
        except Exception as e:
            print(f"Failed: {e}")
