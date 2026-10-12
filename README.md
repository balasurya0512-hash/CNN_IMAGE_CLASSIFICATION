# CNN Leaf Disease Classification

This project classifies leaves into the classes found in the dataset:

- Healthy
- Powdery
- Rust

## Dataset

The dataset is stored in the `Computer vision` folder with one subfolder per class.

## Train the model

```bash
py -3.13 leaf_disease_cnn.py --train
```

## Predict a single image

```bash
py -3.13 leaf_disease_cnn.py --predict "Computer vision\Healthy\sample.jpg"
```

## Run the website

```bash
py -3.13 app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Upload a leaf image and the app will display the predicted disease class.

## Files

- `leaf_disease_cnn.py` - model definition, training, and prediction logic
- `app.py` - Flask website
- `templates/index.html` - upload page
- `leaf_disease_model.keras` - trained model file created after training
- `class_labels.json` - class names saved after training
