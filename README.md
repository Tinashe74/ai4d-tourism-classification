# AI4D Lab Tanzania Tourism Classification

This repository contains the experiments and submission artifacts for the Briisp Academy AI4D Tourism Classification Challenge, the Tanzania tourism expenditure classification problem hosted on Zindi.

The task is six-class classification of a tourist's expenditure range. The target is `cost_category`, and the competition metric is multiclass logarithmic loss (log loss), where lower is better.

The repository contains both the baseline training workflow and a large research history of alternative representations, calibrated ensembles, OOF analyses, specialist models, and submission candidates. The files are intentionally retained so that experiments can be audited and reproduced.

## Dataset

Place the competition data files in the project root:

| File | Purpose |
|---|---|
| `Train.csv` | Training observations and the `cost_category` target. |
| `Test.csv` | Test observations without the target. |
| `SampleSubmission.csv` | Official submission template, including the required ID order and probability-column order. |
| `VariableDefinitions.csv` | Definitions supplied with the dataset. |

The current files contain:

- Training rows: 18,506
- Test rows: 6,169
- Training columns: 21, including the target
- Test columns: 20
- Submission columns: `Tour_ID` plus six probabilities

The six competition classes, in the official submission order, are:

```text
High Cost
Higher Cost
Highest Cost
Low Cost
Lower Cost
Normal Cost
```

The training data include tourist identity/location, demographics, trip purpose and activity, information source, package components, nights spent in mainland Tanzania and Zanzibar, and whether the trip was the tourist's first trip to Tanzania.

`Tour_ID` is an identifier and is not used as an ordinary predictive feature in the baseline feature function. The target column is removed automatically when features are built.

## Installation

### What the repository provides

There is currently no `requirements.txt`, `environment.yml`, `pyproject.toml`, `Pipfile`, or `setup.py` in the repository. A local `.venv` directory is present and was used for the experiments.

The recorded environment is:

```text
Python 3.14.7
numpy 2.5.3
pandas 2.3.3
scikit-learn 1.9.0
scipy 1.18.1
lightgbm 4.7.0
xgboost 3.4.1
catboost 1.2.10
ngboost 0.5.11
country-converter 1.3.2
lifelines 0.30.3
joblib 1.6.0
tqdm 4.70.1
matplotlib 3.11.1
plotly 7.0.0
```

The remaining packages in the local environment are transitive or supporting packages. The exact installed package list can be inspected with:

```powershell
.\.venv\Scripts\python.exe -m pip list --format=freeze
```

### Use the existing environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe --version
```

The project was run with Python 3.14.7. The scripts require the scientific Python stack listed above; because the project does not currently contain a dependency lockfile, package versions should be kept consistent with the recorded environment when reproducing historical results.

### Create a new environment

If the local `.venv` is unavailable, create a replacement environment and install the recorded direct dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy==2.5.3 pandas==2.3.3 scikit-learn==1.9.0 scipy==1.18.1 lightgbm==4.7.0 xgboost==3.4.1 catboost==1.2.10 ngboost==0.5.11 country_converter==1.3.2 lifelines==0.30.3 joblib==1.6.0 tqdm==4.70.1 matplotlib==3.11.1 plotly==7.0.0
```

Package availability can vary by operating system and Python version, especially for compiled libraries such as LightGBM, XGBoost, and CatBoost.

## How to run

All commands below are run from the repository root. The commands use the project environment explicitly so that the active interpreter is unambiguous.

### 1. Prepare the data

Copy `Train.csv`, `Test.csv`, `SampleSubmission.csv`, and `VariableDefinitions.csv` into the project root. The scripts read these exact filenames. No separate download or preprocessing command is implemented in this repository.

Before training, confirm the basic schema:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; print(pd.read_csv('Train.csv').shape); print(pd.read_csv('Test.csv').shape); print(pd.read_csv('SampleSubmission.csv').columns.tolist())"
```

### 2. Run the baseline LightGBM workflow

`train_model.py` is a thin wrapper around `experiment.py`:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

`experiment.py`:

1. Loads `Train.csv` and `Test.csv`.
2. Removes `Tour_ID` and the target from the model feature frame.
3. Converts package Yes/No fields to numeric flags.
4. Adds total nights, party size, package count, Zanzibar share, female share, night ratio, country-purpose, and country-activity features.
5. Factorizes categorical values across the combined train/test feature frame.
6. Uses five-fold stratified cross-validation.
7. Trains three LightGBM multiclass configurations.
8. Scores each configuration with multiclass log loss and weights the test predictions according to the recorded OOF scores.
9. Writes `candidate_submission.csv` with the generated test probabilities.

The baseline is a reference workflow. It is not the final competition result.

### 3. Generate XGBoost OOF predictions

`xgb_compact_oof.py` creates fold-wise out-of-fold predictions and averaged test predictions:

```powershell
.\.venv\Scripts\python.exe xgb_compact_oof.py
```

The default seed is 2026. It writes:

- `xgb_compact_oof.npy`
- `xgb_compact_test.npy`

The script supports the following environment variables:

```powershell
$env:XGB_SEED = '2026'
.\.venv\Scripts\python.exe xgb_compact_oof.py
```

For the optional structural feature set:

```powershell
$env:XGB_STRUCT = '1'
.\.venv\Scripts\python.exe xgb_compact_oof.py
Remove-Item Env:XGB_STRUCT
```

For the optional hybrid interaction feature set:

```powershell
$env:XGB_HYBRID = '1'
.\.venv\Scripts\python.exe xgb_compact_oof.py
Remove-Item Env:XGB_HYBRID
```

The structural and hybrid runs write correspondingly named `.npy` files. These arrays are intermediate model components, not submissions by themselves.

### 4. Run the OOF specialist-model analysis

`xgb_moe_attack.py` reconstructs the validated ensemble baseline from saved OOF components and trains specialist XGBoost models for selected regimes. The regimes used in the successful local-mixture experiment include:

- South Africa plus package tour
- Business or Meetings and Conference purpose
- A long-trip/no-package diagnostic regime

Run it after the required OOF component arrays exist:

```powershell
.\.venv\Scripts\python.exe xgb_moe_attack.py
```

It reports OOF log loss and writes:

- `xgb_moe_results.csv`
- `xgb_moe_specialist_predictions.npy`

This is an analysis step. It does not write the final test submission.

### 5. Generate the validated specialist candidate

`make_xgb_moe_candidate.py` trains the two full-data specialist models, applies the validated 10% membership corrections to the protected champion, checks the official submission schema, and writes:

```powershell
.\.venv\Scripts\python.exe make_xgb_moe_candidate.py
```

Output:

```text
candidate_xgb_moe_southafrica_business_v1.csv
```

The script checks row count, schema, IDs, duplicate IDs, missing values, finite probabilities, probability range, and row sums. It also checks that the protected champion and consensus files are unchanged.

### 6. Generate a controlled tuned candidate

`make_xgb_moe_tuned_candidate.py` is the parameterized version used for controlled weight experiments. It accepts:

- `MOE_OUT`: output filename
- `MOE_W_SA`: South Africa/package weight
- `MOE_W_BIZ`: business/conference weight

Example:

```powershell
$env:MOE_OUT = 'candidate_xgb_moe_sa10_biz30_v6.csv'
$env:MOE_W_SA = '0.10'
$env:MOE_W_BIZ = '0.30'
.\.venv\Scripts\python.exe make_xgb_moe_tuned_candidate.py
Remove-Item Env:MOE_OUT,Env:MOE_W_SA,Env:MOE_W_BIZ
```

The builder uses `SampleSubmission.csv` as the source of the official ID column and row order. It does not overwrite the protected champion or consensus files.

### 7. Generate the adaptive-selector candidate

The adaptive-selector workflow is split into analysis and candidate construction:

```powershell
.\.venv\Scripts\python.exe adaptive_selector_attack.py
```

The analysis script evaluates cross-fitted selector gates and refuses to qualify a selector unless the configured robustness checks pass. `build_adaptive_selector_candidate.py` reconstructs the selector, audits OOF and test gate distributions, checks probability integrity, and writes `candidate_adaptive_selector_v1.csv` only when the checks pass.

This workflow depends on previously generated OOF/test component arrays, including ordinal, hierarchical, local-similarity, calibrated, structural, and hybrid components. It is therefore not a clean-from-zero pipeline unless those component-generation scripts are run first.

### 8. Validate a submission before upload

Every submission must:

- Have 6,169 rows.
- Use the exact ID column and ID order from `SampleSubmission.csv`.
- Use the exact six probability columns in the official order.
- Have no duplicate IDs or missing values.
- Contain finite probabilities in `[0, 1]`.
- Sum to 1 for every row.
- Avoid rounding or thresholding probabilities before upload.

The specialist candidate builders implement these checks. A submission is uploaded manually to Zindi; no script in this repository submits automatically.

## Feature engineering

The main feature function is `experiment.features` in `experiment.py`. It creates:

- Numeric versions of package Yes/No variables.
- `total_nights` from mainland and Zanzibar nights.
- `party_size` from female and male counts.
- `package_count`.
- `zanzibar_share`.
- `female_share`.
- `night_ratio`.
- `country_purpose` interaction.
- `country_activity` interaction.

The XGBoost preparation function in `xgb_compact_oof.py` can additionally create structural features such as nights per person, party per night, Zanzibar share, female share, duration bands, purpose-duration interactions, country-arrangement interactions, country-accommodation interactions, and package signatures when the corresponding flags are enabled.

The repository also contains targeted experiments for frequency encoding, one-hot encoding, target encoding, ordinal modeling, local similarity, duration structure, residual correction, class specialists, calibration, and hidden-structure analysis. These are research artifacts and should be run only after checking their input/output dependencies.

## Important files and folders

### Core data and metadata

- `Train.csv`, `Test.csv`, `SampleSubmission.csv`: competition inputs and official output template.
- `VariableDefinitions.csv`: supplied variable definitions.
- `feature_dictionary.csv`: observed feature types, missingness, and cardinality summary.
- `full_competition_data_audit.csv`, `competition_structure_audit.csv`, and related audit CSVs: dataset and train/test-structure checks.

### Core training code

- `experiment.py`: feature engineering and baseline LightGBM ensemble.
- `train_model.py`: executable wrapper for the baseline workflow.
- `xgb_compact_oof.py`: compact XGBoost five-fold OOF/test component generation.
- `make_xgb_moe_candidate.py`: validated localized specialist candidate builder.
- `make_xgb_moe_tuned_candidate.py`: parameterized localized candidate builder.
- `xgb_moe_attack.py`: OOF analysis for localized mixture-of-experts corrections.
- `adaptive_selector_attack.py`: cross-fitted adaptive-selector analysis.
- `build_adaptive_selector_candidate.py`: adaptive-selector test-time audit and candidate builder.

### Modeling and analysis scripts

The remaining Python files are grouped by experiment family rather than being one linear pipeline:

- `geo_*`, `onehot_*`, `interaction_lgbm.py`, `joint_frequency_lgbm.py`: categorical, geographic, interaction, and frequency representations.
- `catboost_*`, `nativecat_*`: CatBoost/native categorical experiments.
- `ordinal_*`: ordered expenditure-class modeling.
- `hierarchical_*`, `local_similarity_*`, `duration_signal_lgbm.py`: hierarchical, nearest-profile, and trip-duration signals.
- `calibrate_*`, `crossfit_vector_calibration.py`, `final_calibration_check.py`: probability calibration and validation.
- `class_*`, `highest_cost_*`, `residual_*`, `targeted_*`, `boundary_expert_attack.py`: targeted specialists and residual analyses.
- `champion_*`, `final_*`, `master_*`, `validation_*`, `audit_*`: diagnostics, stress tests, validation, and reproducibility audits.
- `make_*` scripts: candidate-generation utilities for specific experiment families.

### Saved model components and logs

- `.npy` files: saved OOF or test probability arrays used as ensemble components.
- `.csv` files beginning with `candidate_`, `submission`, `blend`, `calibrated`, `geo`, `onehot`, `joint`, or `ordinal`: generated predictions or candidate submissions.
- `experiment_run.txt`, `baseline_run.txt`, `catboost_run.txt`, and `ordinal_attack_run.log`: selected run logs.
- `catboost_info/`: CatBoost training metadata and runtime logs.
- `__pycache__/`: Python bytecode cache; it is not part of the modeling method.
- `.venv/`: the local Python virtual environment used for the recorded experiments.
- `solution_reference2/`: present in the workspace but contains no project files in the inspected checkout.

The protected historical files used during the competition include `candidate_xgboost_breakthrough_v3.csv` and `candidate_consensus_correction_final.csv`. Candidate builders are designed to leave them unchanged.

## Results

### Public leaderboard

Log Loss: **1.050298594**

### Private leaderboard

Log Loss: **1.034102889**

### Final ranking

**12th / 60 participants**

**Top 20%**

The private leaderboard used a different portion of the hidden test data from the public leaderboard. The private score therefore determined the final ranking and was not expected to equal the public score.

## Key lessons learned

- Out-of-fold validation was essential for comparing components, estimating log loss, and detecting whether a proposed model added useful information.
- Lower OOF log loss did not always produce a better public leaderboard score. Several apparently stronger blends failed to transfer to the hidden test distribution.
- Probability quality matters because the competition uses log loss. Calibration, temperature experiments, and probability-integrity checks were therefore treated as first-class parts of the workflow.
- Ensemble diversity was useful when the component represented a genuinely different error pattern. Blindly averaging more seeds or adding broad blends could reduce public performance.
- Leakage prevention and cross-fitting mattered for target-dependent selectors and specialist decisions. Selector decisions were evaluated with cross-fitted OOF predictions rather than fitting and evaluating on the same rows.
- Once the score became strong, improvements were small and regime-specific. A localized specialist correction transferred better than repeated broad model changes.
- Public and private leaderboard behavior can differ substantially, so leaderboard improvements were treated as evidence about the hidden split rather than proof of universal generalization.

## Challenges

The main technical challenges were:

- Comparing LightGBM, XGBoost, CatBoost, ordinal models, local-similarity models, neural/tabular experiments, and multiple categorical representations under one log-loss metric.
- Managing the computational cost of five-fold cross-validation, multiple seeds, specialist models, calibration searches, and saved probability arrays.
- Maintaining exact probability validity: official IDs and order, six-class schema, finite values, `[0, 1]` bounds, and row sums of one.
- Distinguishing real ensemble diversity from redundant or unstable predictions.
- Avoiding overfitting to OOF folds, public leaderboard feedback, or a small subgroup while searching for improvements.
- Handling public/private distribution differences and deciding whether a small public improvement was robust enough to justify another submission.
- Preserving reproducibility while retaining a large history of candidate files, intermediate arrays, diagnostics, and run logs.

## Reproducibility

### Validation procedure

- Primary cross-validation: five-fold `StratifiedKFold`.
- Baseline seed: `42` in `experiment.py`.
- Compact XGBoost default seed: `2026`, configurable with `XGB_SEED`.
- XGBoost specialist OOF folds: five-fold stratified CV with fold-dependent model seeds based on `2026`.
- Adaptive-selector stability seeds: `2025`, `2026`, `42`, `777`, and `3407`.
- The adaptive selector uses cross-fitted gate training and a minimum champion/base weight; its candidate builder also compares OOF and test-time gate distributions.

### Modeling configuration

The baseline trains three LightGBM multiclass configurations and blends them using OOF log-loss-derived weights. The compact XGBoost workflow uses histogram tree training, multiclass soft probabilities, early stopping, and engineered numeric/categorical representations. The validated localized specialist uses XGBoost models with the following principal settings:

```text
n_estimators=260
max_depth=3
learning_rate=0.055
min_child_weight=8
subsample=0.90
colsample_bytree=0.90
reg_alpha=0.2
reg_lambda=10
tree_method=hist
objective=multi:softprob
num_class=6
```

The exact candidate-generation scripts should be treated as the source of truth for any specific historical candidate. Saved `.npy` arrays are intermediate predictions and must not be mixed across incompatible preprocessing or fold definitions.

## Author

**Tinashe Victor Banda**  
Information and Communications University (ICU)  
Bachelor of Information Communication and Technology

University Hackathon participant.

## Acknowledgements

Acknowledgement is given to:

- Briisp Academy
- AI4D
- Zindi

The project uses the Tanzania tourism classification dataset supplied for the competition. The repository contains variable definitions and competition data files, but does not include a separate official Tanzania National Bureau of Statistics citation; no additional institutional attribution is asserted here beyond the competition materials.
