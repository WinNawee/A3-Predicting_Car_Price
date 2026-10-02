# A3: Predicting Car Price — Classification

AT82.03 Machine Learning — Assignment 3. Continues from Assignment 1/2's
used-car price dataset, now treated as a 4-class classification problem:
`selling_price` is bucketed into 4 price bands, and a multinomial
Logistic Regression built from scratch (softmax, cross-entropy loss,
gradient descent, optional Ridge/L2 penalty) predicts which band a car
falls into.

**Live site**: [https://web-st127031.ml.brain.cs.ait.ac.th](https://web-st127031.ml.brain.cs.ait.ac.th)

## Repository contents

```
.
├── README.md
├── car_price_classification_A3.ipynb   # Task 1 + Task 2 + Task 3 (MLflow), full notebook
├── data/
│   └── Cars.csv                         # raw A1/A2 Car Price dataset
├── result/                               # MLflow screenshots (runs, best run, model registry)
└── app/                                  # Task 3 web application
    ├── Dockerfile
    ├── docker-compose.yaml
    ├── tests/
    │   └── test_model.py                 # unit tests (Task 3, Objective 3)
    └── code/
        ├── app.py                        # entry point, shared nav bar, page routing
        ├── model_inference.py            # shared predict() used by the app + tests
        ├── requirements.txt
        ├── assets/
        │   └── style.css
        ├── pages/
        │   ├── home.py                   # landing page, compares the A1, A2 and A3 models
        │   ├── old_model.py              # /old     — A1 model (scikit-learn pipeline), unchanged from A2
        │   ├── new_model.py              # /new     — A2 model (from-scratch linear regression), unchanged from A2
        │   └── predict.py                # /predict — A3 classifier (this assignment)
        └── model/
            ├── car_price_classifier.pkl  # A3: W (weight matrix) + prep (ColumnTransformer) + metadata
            ├── car_price_model.pkl       # A1: scikit-learn pipeline
            ├── car_price_model_v2.pkl    # A2: theta + prep + residual std
            └── model_metadata.pkl        # A1/A2: shared owner scale + column lists
```

## Task 1 — Classification

- `selling_price` → 4 ordered classes (**Budget / Mid-range / Premium /
  Luxury**) via `pd.qcut` (quantile cut), bin edges learned from the
  training split only, giving ~25% of samples per class.
- `LogisticRegression` (based on `02 - Multinomial Logistic
  Regression.ipynb`): softmax output, cross-entropy loss, batch /
  minibatch / stochastic gradient descent. The gradient is normalized by
  batch size `m` (the textbook version isn't, which only stays stable on
  tiny datasets — see the notebook's Section 3.1 for why that matters
  here).
- `accuracy`, per-class `precision` / `recall` / `f1_score`, and their
  `macro_*` / `weighted_*` averages — all implemented from scratch (no
  `sklearn.metrics` calls inside them) and cross-checked against
  `sklearn.metrics.classification_report` on intentionally imbalanced
  mock data (Section 3.3 of the notebook).
- **What does `support` mean?** The number of *true* samples in each
  class — independent of the model's predictions, and exactly the weight
  `weighted avg` uses.

Preprocessing follows the same A1/A2 conventions: `mileage`/`engine`/
`max_power` parsed out of their unit strings, `brand` extracted from
`name` (rare brands grouped into `"Other"`), `owner` ordinally encoded,
missing values imputed via `sklearn.impute.SimpleImputer` fit on the
training split only, `fuel` kept to Diesel/Petrol (CNG/LPG are a ~1%
minority with a different mileage unit, excluded upstream same as A1/A2).

### Avoiding data leakage

- **Duplicates removed before splitting.** The raw data repeats 1,220
  listings (identical on every column the model sees). Left in, ~20% of
  the test set had an exact copy in train, inflating test scores. After
  `drop_duplicates`, train/test overlap is 0 (checked in the notebook).
- **Train / validation / test = 64 / 16 / 20.** Brand grouping, price
  bucket edges, imputers and scaler are fit on **train only**.
- **Model selection uses validation only** — the no-penalty vs. ridge
  comparison and the MLflow grid are all scored on validation, and the
  best run is chosen by validation `macro_f1`.
- **The test set is scored exactly once**, on the already-chosen model.

| Best model (batch GD, α=1.0, ridge λ=0.5) | Validation | Test |
|---|---|---|
| Accuracy | 0.744 | 0.730 |
| Macro F1 | 0.738 | 0.726 |
| Weighted F1 | 0.746 | 0.733 |

## Task 2 — Ridge Logistic Regression

$$ J(\theta) = -\sum_{i=1}^m y^{(i)}\log(h^{(i)}) + \lambda\sum_{j=1}^n \theta_j^2 $$

`RidgePenalty` (adapted from `03 - Bias-Variance Tradeoff and
Regularization.ipynb`) applies this to the weight **matrix** of the
multinomial model (excluding the bias row). Toggle with
`LogisticRegression(..., use_penalty=True, l=<lambda>)`.

## Task 3 — Deployment

> **Note on the course MLflow server.** Per the TA announcements
> (21 Sep and 1 Oct 2026), the course MLflow server
> (`mlflow.ml.brain.cs.ait.ac.th`) was down / unstable, so logging to it
> and registering on it (Objectives 1 and 2) are no longer required.
> Experiments were instead logged to a **local MLflow instance**
> (`sqlite:///mlflow_fallback.db`, as in A2), and the screenshots below
> are submitted in place of the server logs. The notebook still tries
> the course server first and falls back to the local store
> automatically, so it will log there unchanged once the server is back.

### 1. MLflow experiment logging

Experiment `st127031-a3`, 5 runs (gradient-descent method × learning
rate × ridge on/off), each logging params + **validation** metrics, with
the model saved as an artifact and no dataset logged. Runs are sorted by
`val_macro_f1`; the best one is `batch-alpha1.0-penaltyTrue-l0.5`.

![MLflow runs](result/mlflow_runs.png)

The chosen run additionally carries its one-time `test_*` metrics
(test accuracy 0.730, test macro F1 0.726):

![Best run metrics](result/mlflow_best_run.png)

### 2. Model registry

The run with the best **validation** `macro_f1` is registered as
`st127031-a3-model` (version 1) and transitioned to **Staging**
(Section 7 of the notebook does this with `MlflowClient`; the UI
equivalent is run → "Register Model" → Models tab → Stage → Transition
to Staging).

![Model registry: st127031-a3-model, Staging](result/mlflow_model_registry.png)

The model is logged as a small MLflow **pyfunc** wrapping just `prep`
(the fitted `ColumnTransformer`) and `W` (a plain numpy weight matrix) —
not the custom `LogisticRegression` class instance. The deployed app
loads the same two objects directly (`app/code/model_inference.py`), so
nothing notebook-specific needs to be importable at serving time.

### 3. CI/CD

- `app/tests/test_model.py` — two unit tests, run with `pytest`:
  1. `test_model_accepts_expected_input` — accepts correctly-shaped
     input (including missing values the imputers handle, and an
     unseen brand `OneHotEncoder` was fit to ignore), and raises on a
     genuinely malformed input (a missing required column).
  2. `test_model_output_shape` — predictions are `(m,)` in `{0,...,k-1}`;
     probabilities are `(m, k)` with rows summing to 1.
- `.github/workflows/ci-cd.yml`: a `test` job runs `pytest` on every
  push; a `deploy` job (on `main`, gated on `needs: test`) builds the
  Dash app's Docker image and pushes it to GitHub Container Registry. A
  Docker Hub push + SSH-deploy-to-VM step (matching the
  `docker-compose.yaml` pull-based deployment below) is included but
  commented out, since it needs repo secrets (`DOCKERHUB_*`, `VM_*`) that
  only exist once you've set up that VM/registry yourself.

## Running it yourself

**Notebook:**
```bash
pip install -r app/code/requirements.txt jupyter matplotlib seaborn
jupyter notebook car_price_classification_A3.ipynb
```

**Web app (locally):**
```bash
cd app/code
pip install -r requirements.txt
python app.py
```
Open `http://localhost:8050`.

The site keeps the A1 and A2 models from the previous deployment and adds
the A3 classifier, all behind one nav bar:

| Page | Model | Output |
|---|---|---|
| `/` | — | compares the three models and lists the A3 price classes |
| `/old` | A1 scikit-learn pipeline | a price |
| `/new` | A2 from-scratch linear regression | a price + ~90% range |
| `/predict` | **A3 from-scratch multinomial logistic regression** | a price class + probability for each class |

**Web app (Docker):**
```bash
cd app
docker build -t car-price-classifier .
docker run -p 8050:8050 car-price-classifier
```

The app serves the model file produced by the notebook
(`app/code/model/car_price_classifier.pkl`). Because the course MLflow
server is unstable, loading from the registry
(`models:/st127031-a3-model/Staging`) is opt-in: set
`USE_MLFLOW_REGISTRY=1` (it uses a 5-second timeout and falls back to the
local file if the registry is unreachable, so a hung server can't keep
the site from starting).

**Tests:**
```bash
pip install -r app/code/requirements.txt
pytest app/tests/ -v
```

### Deployment

Deployed the same way as A2: build the image, push it to Docker Hub
(`naweep/car-price-classifier:latest`), and let `docker-compose.yaml` on
the course's `ml-brain` server pull it. The compose file joins the
existing `web` Docker network and adds Traefik routing labels
(`entrypoints=websecure`, `certresolver=letsencrypt`) — no ports are
published directly; Traefik reaches the container over that network.
It uses the same host as the A2 deployment
(`web-st127031.ml.brain.cs.ait.ac.th`), so the A2 container is stopped
before this one is started.