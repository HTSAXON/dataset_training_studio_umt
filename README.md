# Image Dataset Petri Net Training Simulation

This browser-based application demonstrates real-time neural-network training and prediction for image classification datasets.

## What it does

- Uses the built-in Fashion-MNIST demo dataset, or any local image dataset arranged in class folders.
- Trains a multi-layer perceptron on resized 28x28 grayscale images.
- Shows live mini-batch images currently being used for training.
- Updates live loss and batch accuracy during training, not only after each epoch.
- Loads a random test image and predicts its class.
- Simulates a Petri net flow:
  - `Idle`
  - `DatasetReady`
  - `Training`
  - `ModelReady`
  - `SampleLoaded`
  - `Predicting`
  - `Classified`
  - `Result`

## Custom Dataset Format

Put your dataset in folders where each class has its own folder:

```text
data/custom/flowers/
  rose/
    image1.jpg
    image2.jpg
  sunflower/
    image1.jpg
    image2.jpg
  tulip/
    image1.jpg
    image2.jpg
```

You can also provide separate train and test folders:

```text
data/custom/flowers/
  train/
    rose/
    sunflower/
  test/
    rose/
    sunflower/
```

Supported image types:

```text
.jpg, .jpeg, .png, .bmp, .webp
```

## Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the application:

```powershell
python app.py
```

If `python` is not on PATH, run it with the bundled runtime:

```powershell
& 'C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Dell\OneDrive\Documents\New project 6\app.py'
```

The app opens in your browser at:

```text
http://127.0.0.1:8765
```

Inside the application:

- For the demo, click `Use Fashion-MNIST Demo`.
- For your own dataset, type a folder path like `data/custom/flowers`, then click `Load Folder Dataset`.
- Click `Start Training`.
- Watch live batch images, loss, batch accuracy, and training curves.
- Click `Load Random Test Image`.
- Click `Predict Sample`.

## Project Structure

- `app.py`: local web server, browser UI, Petri net simulation, and training visualization.
- `fashion_pipeline.py`: generic image dataset loading, demo dataset download, training, and prediction helpers.
- `scripts/download_dataset.py`: downloads the Fashion-MNIST demo files.
- `scripts/train_model.py`: optional command-line training helper.
