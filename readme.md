# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link

---

## Candidate Run Instructions (Santosh Adapa)

### 1. Environment Setup
To run the code, ensure you have Python 3.9+ installed. Install the dependencies listed in the `requirements.txt` file:
```bash
python -m pip install -r requirements.txt
```

### 2. Running the Training & Prediction Pipeline
The entire machine learning pipeline (Target Encoding, Feature Engineering, LightGBM Ensemble Training, and Inference) is contained within a single executable script:
```bash
python train.py
```
This script will automatically output both `validation_predictions.csv` and `december-chart-inputs.csv` directly into the root directory.

### 3. Evaluating the Output
Once the pipeline finishes, run the provided evaluation script to generate the final chart:
```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```
