# VisTAD: Vision-Language Model Guided Knowledge Distillation for Time-Series Anomaly Detection

## Environment Setup

```bash
pip install -r requirements.txt
```
---

## Reproducing Results

```bash
python reproduce_results.py
```

---

## Train Student Models

```bash
# 1. SMD: Pooled 28-Server Foundation Model (1D-CNN, 25 epochs)
python train.py --config configs/smd.yaml --dataset smd --entity all --arch cnn_1d --num_layers 3 --epochs 25 --batch_size 128 --tag foundation_pooled28

# 2. SWaT: Deep Residual TCN (5 Layers, 25 epochs)
python train.py --config configs/swat.yaml --dataset swat --entity swat_attack --arch tcn --num_layers 5 --epochs 25 --batch_size 64 --tag swat_tcn5

# 3. GECCO: Water Quality 1D-CNN (25 epochs, tau=0.50)
python train.py --config configs/gecco.yaml --dataset gecco --entity gecco_water --arch cnn_1d --num_layers 3 --epochs 25 --batch_size 64 --decision_threshold 0.50 --tag gecco_cnn3

# 4. MSL: Spacecraft Telemetry Residual TCN (30 epochs)
python train.py --config configs/msl.yaml --dataset msl --entity msl_test --arch tcn --num_layers 5 --epochs 30 --batch_size 64 --w_sup 1.5 --w_feat 0.05 --w_resp 0.2 --pos_weight 4.0 --decision_threshold 0.68 --tag msl_tcn5
```

---

## Launching the Demo App

A live interactive test interface is provided as ran below:

```bash
cd streamlit_app
streamlit run app.py
```

*What to do in the demo?*
- **Domain Selector:** Choose between SMD, SWaT, MSL, or GECCO.
- **Threshold Calibration Slider:** Tune your decision threshold for the scor given by the models 
- **Select smoothing parameters**: You can also choose the min continuous anomaly score to reduce false positives, and you can club windows to ensure smooth predictions.

---

## Running as a Notebook (on Kaggle)

We can run the notebook `vistad_kaggle_runner.ipynb`. For generating caches, and training the initial student models. While we can run `kaggle_experiments.ipynb` for running ablations to compare models under various configurations.
