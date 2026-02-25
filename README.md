# GreenGuard (EECS 159)

Smart irrigation web app with:
- Arduino sensor ingest (`temp.ino`)
- Flask backend + SQLite (`app.py`, `temp.db`)
- Authenticated dashboard pages (`static/`)
- Image-based watering recommendation (`/ai-check`)

## Features

- User register/login (session-based auth)
- Calendar view of watered vs non-watered days
- Sensor chart view (time-series data from DB)
- Current watering state indicator
- Auto-watering logic on low soil moisture
- AI image upload page for `WATER` / `DO_NOT_WATER` prediction
- Fake Arduino simulator for local testing (`fake_arduino.py`)

## Project Structure

```text
app.py                       # Main Flask app
fake_arduino.py              # Arduino behavior simulator
temp.ino                     # Real Arduino sketch
train_soil_classifier.py     # Model training script
temp.db                      # SQLite database
models/soil_water_classifier.pt
static/
  auth.html
  index.html                 # Calendar page
  sensor.html                # Chart page
  ai_check.html              # Image inference page
dataset/
  train/
    WATER/
    DO_NOT_WATER/
  val/
    WATER/
    DO_NOT_WATER/
```

## Requirements

- Python 3.10+
- Packages:
  - `flask`
  - `werkzeug`
  - `pillow`
  - `torch`
  - `torchvision`

Install:

```bash
pip install flask werkzeug pillow torch torchvision
```

## Run the App

```bash
python app.py
```

Open:
- `http://localhost:5000/login`

## Fake Arduino Testing

Run simulator (continuous loop, every 2 seconds by default):

```bash
python fake_arduino.py --server http://127.0.0.1:5000 --username test
```

Run one cycle only:

```bash
python fake_arduino.py --server http://127.0.0.1:5000 --username test --once
```

## Model Training

Train from `dataset/train` and validate on `dataset/val`:

```bash
python train_soil_classifier.py
```

Output model:
- `models/soil_water_classifier.pt`

`app.py` loads this model in `/api/ai-water-check`.

## Dataset Attribution

Soil moisture image dataset used for training:

- Mendeley Data: https://data.mendeley.com/datasets/skcc44yvvg/2

Current label mapping used in this project:
- `dry` -> `WATER`
- `moderate` + `wet` -> `DO_NOT_WATER`

## Notes

- Auto-watering command logic is controlled in `app.py` constants:
  - `SOIL_DRY_TRIGGER_PERCENT`
  - `WATERING_DURATION_SECONDS`
  - `MIN_SECONDS_BETWEEN_WATERING`
- Real-time status guards stale sensor rows using:
  - `REALTIME_SENSOR_MAX_AGE_SECONDS`
