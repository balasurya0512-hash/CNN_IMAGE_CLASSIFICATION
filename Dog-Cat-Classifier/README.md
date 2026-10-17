# Cat and Dog RNN Classifier

This project trains a bidirectional LSTM on the images in `Data/Train_Data`.
Each image is resized to `96 x 96` pixels and interpreted as a sequence of 96 image rows. The 3 RGB channels in each row form the features for the RNN.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

Run this from the `Dog-Cat-Classifier` directory:

```powershell
python train_rnn.py --epochs 15
```

The script reserves 20% of the images for validation and saves the best model as `cat_dog_rnn.keras`.

## Predict sample images

```powershell
python train_rnn.py --epochs 15 --predict test_cat.jpg test_dog.jpg
```

The prediction images are evaluated after training. The output label is `cat` or `dog` with the model confidence.