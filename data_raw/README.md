# data_raw/ — source videos live here

```
data_raw/
├── real/   # genuine videos (or images)
└── fake/   # manipulated videos (or images)
```

Nothing in this folder is committed to git. You can drop files in by hand, or let
`scripts/import_dataset.py` sort a downloaded dataset for you (recommended).

## Fastest route: Kaggle → import script

1. Free Kaggle account → download one of these:

| Dataset | Kaggle location | Size | Notes |
|---|---|---|---|
| **Celeb-DF v2** (recommended) | `reubensuju/celeb-df-v2` | ~10 GB | 890 real (Celeb-real + YouTube-real), 5,639 fake (Celeb-synthesis). High-quality swaps; the standard "hard" benchmark. |
| **DFDC sample** | competition `deepfake-detection-challenge`, file `train_sample_videos.zip` | ~4 GB | 400 videos with `metadata.json` labels. Accept the competition rules once before downloading. |
| FaceForensics++ | not on Kaggle — request via the [official form](https://github.com/ondyari/FaceForensics) | varies | 1,000 real + 4,000 fake across four methods. Use **c23**. Approval can take a few days. |

2. Sort into real/fake with a balanced cap per class:

```bash
python scripts/import_dataset.py --source celebdf /path/to/celeb-df-v2 --limit 300
python scripts/import_dataset.py --source dfdc    /path/to/train_sample_videos --limit 200
python scripts/import_dataset.py --source ffpp    /path/to/FaceForensics++ --limit 300
python scripts/import_dataset.py --source folders /path/real /path/fake      # anything else
```

Add `--move` to move instead of copy (saves disk space).

3. Then:

```bash
python prepare_dataset.py            # -> dataset/{train,validation,test}/{real,fake}/ + manifest.csv
python train.py
python evaluate.py --gradcam-samples 6
python app.py
```

## No GPU?

Use `notebooks/train_colab.ipynb` on Google Colab (free T4). It does the Kaggle download,
import, extraction, training and evaluation for you and gives back a `trained_model.zip`.

## How much data?

For a minor project, ~150–300 videos per class is plenty: with 16 frames per video that is
5–10 k face crops, which trains in ~15 minutes on a T4 and 1–2 hours on a laptop CPU
(use `--architecture mobilenetv2` on CPU).
