# Food Quality Detection

A Streamlit demo that classifies an uploaded or camera-captured fruit image as **Fresh** or **Rotten** using a fine-tuned EfficientNet-B0 model. It can also display the saved training accuracy and loss curves.

> This is an educational image-classification demo, not a food-safety assessment. Do not use its predictions to decide whether food is safe to eat.

## Repository contents

- `app.py` — Streamlit application
- `model_2_1_26_2.h5` — trained model loaded by the application
- `history_combined.pkl` — saved training history used for the optional charts
- `Final_ml_project.ipynb` — model-training notebook
- `REPORT_ML_PROJECT.pdf` — project report
- `requirements.txt` — runtime dependencies

## Run locally

Use Python 3.10 or 3.11. Create and activate a virtual environment:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
```

Install dependencies and start the app from the repository root:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Upload a JPG, JPEG, or PNG image, or use the camera input, then select **Run Quality Analysis**.

## Model and files

The app expects `model_2_1_26_2.h5` in the repository root. Training charts are shown when `history_combined.pkl` is present and contains the two training phases (`h1` and `h2`). Confusion-matrix and classification-report images are optional; the app skips them when they are not present.

The model was trained for fruit freshness classification. Results can be affected by lighting, image framing, fruit type, and differences between the training data and the supplied image. Confidence is the model's output, not a calibrated probability or guarantee of correctness.

## Troubleshooting

- If the model fails to load, check that the model file is present and that the TensorFlow version from `requirements.txt` is installed in the active environment.
- If installation fails, use Python 3.10 or 3.11 and install into a fresh virtual environment.
- Large model files may make cloning or deployment slower.
