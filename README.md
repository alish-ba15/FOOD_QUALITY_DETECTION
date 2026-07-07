# FOOD QUALITY DETECTION SYSYTEM
FOOD QUALITY DETECTION SYSYTEM:
This project classifies fruits as Fresh or Rotten using EfficientNet-B0. It includes the trained model and a Streamlit web interface.

# Project Structure
app2.py - Streamlit web application
2_1_26.h5 - Trained EfficientNet-B0 model
history_combined.pkl - Saved training history (accuracy and loss)
Final_ml_project.ipynb - Notebook used for training

# Setup and Run
Make sure Python is installed. It is recommended to use a virtual environment.

# Create and activate a virtual environment:
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate

# Install the required libraries:
pip install streamlit tensorflow numpy pillow

# Run the Streamlit app:
streamlit run app2.py
Make sure the model file model_2_1_26_1.h5 is in the same folder as app2.py.

# How it Works:
Images are preprocessed and augmented to improve model robustness. EfficientNet-B0 is used with fine-tuning, where the last 20 layers are unfrozen to specialize the model for fruit textures. The Streamlit app takes an uploaded image and predicts whether it is Fresh or Rotten along with a confidence score.

# Evaluation:
Training history in combined.pkl shows high accuracy. Performance can be evaluated using the confusion matrix in the notebook.

