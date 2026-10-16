# data_raw/ — put your source videos here

```
data_raw/
├── real/   # genuine videos (or images)
└── fake/   # manipulated videos (or images)
```

Sub-folders are fine (`data_raw/fake/Deepfakes/…`, `data_raw/fake/Face2Face/…`);
`prepare_dataset.py` searches recursively. Nothing in this folder is committed to git.

## Where to get data

| Dataset | Size | Notes |
|---|---|---|
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | 1,000 real + 4,000 fake videos | Request access via the form in the repo. Use the **c23** compression level. The four manipulation types (Deepfakes, Face2Face, FaceSwap, NeuralTextures) can all go under `fake/`. |
| [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) | 590 real + 5,639 fake | Harder, higher-quality fakes. Good as a **cross-dataset test** for a model trained on FF++. |
| [DFDC sample](https://www.kaggle.com/c/deepfake-detection-challenge/data) | 400 videos (train sample) | Kaggle login required. Labels are in `metadata.json`; split the files into `real/` and `fake/` with a small script. |

For a minor project, ~150–300 videos per class is plenty: with 16 frames per
video that is 5–10 k face crops, which trains in well under an hour on a
laptop GPU (or a few hours on CPU with `--architecture mobilenetv2`).

Then:

```bash
python prepare_dataset.py            # -> dataset/{train,validation,test}/{real,fake}/ + manifest.csv
python train.py
python evaluate.py --gradcam-samples 6
python app.py
```
