# Wine ML Classifier

Simple machine learning pipeline for classifying wine varieties using Python and scikit-learn.

This repository demonstrates a minimal and reproducible ML workflow including:
- dataset loading
- preprocessing
- model training
- evaluation

The project is intended for educational purposes and as a minimal example of a machine learning experiment repository.


## Dataset

This project uses the Wine dataset, a classic dataset in machine learning used for classification tasks.

Dataset characteristics:

- Samples: 178
- Features: 13 numerical chemical analysis measurements of wines
- Classes:
  - class_0
  - class_1
  - class_2

The features correspond to chemical properties of wine such as:

- alcohol
- malic acid
- ash
- alcalinity of ash
- magnesium
- total phenols
- flavanoids
- nonflavanoid phenols
- proanthocyanins
- color intensity
- hue
- OD280/OD315 of diluted wines
- proline

The goal is to predict the wine class based on these chemical measurements.

## Installation

Clone the repository:

```bash
git clone https://github.com/andredemedeiros/wine-ml-classifier.git
cd wine-ml-classifier
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset download

Run the script:

```bash
python data/data_download.py
```

## Models training and validation

Run the script:

```bash
python model.py
```

## Model Ranking Results
**135 combinations tested** — 9 preprocessors × 15 models, evaluated with 5-fold CV + held-out test set.

### 🥇 Best Combination

| Property | Value |
|---|---|
| **Preprocessor** | StandardScaler |
| **Model** | ExtraTrees |
| **Test Accuracy** | `0.6906` |
| **Test F1 (weighted)** | `0.6750` |
| **CV Mean ± Std** | `0.6685 ± 0.0157` |

## License

This project is licensed under the MIT License.