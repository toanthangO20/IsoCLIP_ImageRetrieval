# IsoCLIP Image-to-Image Retrieval

This project is a trimmed split of the original IsoCLIP repository. It keeps the
image-to-image retrieval pipeline and the Kaggle CUB-200-2011 notebook only.

## Kept

- `kaggle_isoclip_cub2011_pipeline.ipynb`
- `src/retrieval.py`
- IsoCLIP/CLIP feature helpers in `src/`
- Image retrieval dataset loaders in `src/datasets/`
- Image-to-image experiment launchers in `exp_img-img_retrieval/`
- `aggregate_retrieval.py`

## Removed From This Split

- Image classification experiment code
- Text-to-text retrieval experiment code
- Paper/full benchmark notebooks unrelated to the CUB Kaggle pipeline
- Original assets and local dataset caches

## Kaggle CUB Pipeline

Open `kaggle_isoclip_cub2011_pipeline.ipynb` on Kaggle. The notebook installs
its own runtime dependencies, downloads or reuses CUB-200-2011, validates the
CUB loader, and runs:

- baseline CLIP image-to-image retrieval
- IsoCLIP image-to-image retrieval

## Local Setup

```bash
bash install_requirements.sh
```

Then run retrieval from the project root, for example:

```bash
python src/retrieval.py \
  --dataroot /path/to/datasets \
  --dataset_name cub2011 \
  --clip_model_name ViT-B/32 \
  --query_eval_type image \
  --gallery_eval_type image \
  --out_path local_cub2011
```
