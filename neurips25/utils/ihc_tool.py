import os 
import torch 
from PIL import Image
import geopandas as gpd
from IPython.display import display
from huggingface_hub import snapshot_download

from trident import OpenSlideWSI, ImageWSI
from trident.segmentation_models import segmentation_model_factory

from omegaconf import OmegaConf

conf = OmegaConf.load("neurips25/configs/base.yaml")


# a. Download a WSI
OUTPUT_DIR = conf.hancock.block1unifeatures
DEVICE = f"cuda:0" if torch.cuda.is_available() else "cpu"

TARGET_MAG = 20
PATCH_SIZE = 256

from trident.patch_encoder_models import encoder_factory

PATCH_ENCODER = "uni_v2" # Visit the factory or check the README for a list of all available models

# a. Instantiate UNI model using the factory 
encoder = encoder_factory(PATCH_ENCODER)
encoder.eval()
encoder.to(DEVICE)
# b. Run UNI feature extraction
features_dir = os.path.join(OUTPUT_DIR, f"features_{PATCH_ENCODER}")

# a. Download a WSI
OUTPUT_DIR = conf.hancock.block1unifeatures
DEVICE = f"cuda:0" if torch.cuda.is_available() else "cpu"

for idx, WSI_FNAME in enumerate(os.listdir(conf.hancock.block1unifeatures)):
    print(idx)
    if WSI_FNAME.endswith(".csv"):
        continue
    if WSI_FNAME.replace("png", "h5") in os.listdir(conf.hancock.block1unifeatures + "/features_uni_v2"):
        print(f"Already processed {WSI_FNAME}")
        continue

    # b. Create ImageSlideWSI
    wsi_path = os.path.join(conf.hancock.block1unifeatures, WSI_FNAME)
    slide = ImageWSI(slide_path=wsi_path, lazy_init=False, mpp=1)

    # c. Run segmentation 
    segmentation_model = segmentation_model_factory("hest")
    geojson_contours = slide.segment_tissue(segmentation_model=segmentation_model, target_mag=10, job_dir=OUTPUT_DIR, device=DEVICE)

    coords_path = slide.extract_tissue_coords(
        target_mag=TARGET_MAG,
        patch_size=PATCH_SIZE,
        save_coords=OUTPUT_DIR
    )
    feats_path = slide.extract_patch_features(
        patch_encoder=encoder,
        coords_path=coords_path,
        save_features=features_dir,
        device=DEVICE
    )
