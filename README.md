# Credit default probability

Score the probability that a credit-card client defaults next month, and serve that score over HTTP.

## Problem

A lender wants a probability of default for each client before the next payment cycle. The positive class is default (`1`). A missed default (false negative) is treated as more costly than a false alarm (false positive).

The model is chosen by validation ROC-AUC, which measures ranking and does not depend on a cutoff. The operating threshold is chosen separately on the validation fold by minimizing `5 * FN + 1 * FP`. PR-AUC and F1 are reported at that threshold. The test fold is scored once, after the model, the configuration, and the threshold are frozen.

## Data

Source: [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), UCI Machine Learning Repository, donated by I-Cheng Yeh.

- Citation: Yeh, I-Cheng (2009). Default of Credit Card Clients. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
- Paper: Yeh, I-Cheng, and Lien, Che-hui (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480.
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Shape: 30,000 clients, 23 explanatory columns, one binary target (`default payment next month`). No missing cells and no duplicate ids.
- Default rate: 6,636 / 30,000 = 22.12%.
- Window: repayment status, bill amounts, and payments from April 2005 through September 2005. The target is default in the following month. Amounts are New Taiwan dollars.

`src/data.py` downloads the official zip and checks the SHA-256 of the zip and of the workbook. A changed upstream file stops the loader instead of silently training on something else. The raw file is written to `data/raw/` and is not committed.

Quality issues in the published file:

- Class imbalance: 22.12% default.
- Education codes `0`, `5`, and `6` are outside the documented set `{1, 2, 3, 4}` (345 rows: 14 + 280 + 51). They are mapped to `4` (other). Documented counts: graduate school 10,585, university 14,030, high school 4,917, other 123.
- Marriage code `0` is outside `{1, 2, 3}` (54 rows) and is mapped to `3` (other). Documented counts: married 13,659, single 15,964, other 323.
- 1,930 clients have a negative bill in at least one month (3,932 negative cells). Those values are kept.
- Payment amounts have a long right tail. The maximum cell is 1,684,259; the 99th percentile of payment cells is about 70,181. Those values are kept.
- `SEX` is on the file (1 = male, 2 = female) and is not used as a feature.

The source names the September 2005 repayment status `PAY_0`. There is no `PAY_1` column. `PAY_2` through `PAY_6` run from August back to April. Status values in the file run from -2 to 8. The documentation describes -1 as pay duly and 1..8 as months of delay. `-2` and `0` are present and are kept as numeric values.

| Field | Meaning |
| --- | --- |
| `limit_bal` | Credit limit, NT dollar |
| `education` | 1 graduate school, 2 university, 3 high school, 4 other |
| `marriage` | 1 married, 2 single, 3 other |
| `age` | Age in years (21–79 in this file) |
| `pay_0` | Repayment status, September 2005 |
| `pay_2` | Repayment status, August 2005 |
| `pay_3` | Repayment status, July 2005 |
| `pay_4` | Repayment status, June 2005 |
| `pay_5` | Repayment status, May 2005 |
| `pay_6` | Repayment status, April 2005 |
| `bill_amt1` … `bill_amt6` | Bill statement, September back to April |
| `pay_amt1` … `pay_amt6` | Amount paid, September back to April |

## Approach

Stratified split, seed 42: 21,000 train / 4,500 validation / 4,500 test. Each validation and test fold has 995 defaults (22.11%). The training fold has 4,646 defaults and 16,354 non-defaults (22.12%).

Cleaning maps the undocumented education and marriage codes above. No rows are dropped. `id` and `sex` are excluded. Education and marriage stay as integers. Trees can split on those codes directly.

Baseline: logistic regression, L2 (`l1_ratio=0`), `class_weight="balanced"`, one-hot encoding for education and marriage, standard scaling for the other inputs.

Selected model: LightGBM, binary objective, `scale_pos_weight = 16354 / 4646 = 3.520` computed on the training fold only. Three configurations were fit with early stopping on validation ROC-AUC (patience 40, learning rate 0.05, up to 400 trees):

| `num_leaves` | `min_child_samples` | Validation ROC-AUC | Trees kept |
| --- | ---: | ---: | ---: |
| 15 | 40 | 0.7750 | 11 |
| 31 | 50 | 0.7817 | 13 |
| 63 | 100 | 0.7811 | 13 |

The served model is the middle row. The text booster is `models/model.txt`. `models/metadata.json` stores the threshold, the training contract, and the metrics. `models/baseline.joblib` and `models/split_ids.json` exist so `python -m src.evaluate` scores the same frozen test ids.

For each model, the threshold minimizes `5 * FN + 1 * FP` on validation probabilities. Ties prefer higher recall, then the lower threshold. LightGBM lands at 0.29. Logistic regression lands at 0.51. Feature ranking uses LightGBM gain. The top shares are `pay_0` 0.588, `pay_amt2` 0.070, `pay_2` 0.051, `limit_bal` 0.041, `pay_4` 0.035.

## Results

Validation was used to pick the LightGBM configuration and both thresholds. Test was not.

| Model | Split | ROC-AUC | PR-AUC | F1 | Precision | Recall | Threshold | Cost `5·FN+FP` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | validation | 0.7311 | 0.5172 | 0.5018 | 0.4219 | 0.6191 | 0.51 | 2739 |
| LightGBM | validation | 0.7817 | 0.5560 | 0.4799 | 0.3423 | 0.8020 | 0.29 | 2518 |
| Logistic regression | test | 0.7117 | 0.4989 | 0.4754 | 0.3911 | 0.6060 | 0.51 | 2899 |
| LightGBM | test | 0.7789 | 0.5402 | 0.4749 | 0.3364 | 0.8070 | 0.29 | 2544 |

On the test fold LightGBM leads by 0.0672 ROC-AUC and 0.0413 PR-AUC. Test F1 is 0.4749 versus 0.4754. The cost rule spends precision to catch more defaults: LightGBM recall is 0.8070 versus 0.6060, and the test cost is 2,544 versus 2,899.

Test confusion matrices at those frozen thresholds. Rows are actual non-default then default. Columns are predicted non-default then default.

Logistic regression:

```
[[2566  939]
 [ 392  603]]
```

LightGBM:

```
[[1921 1584]
 [ 192  803]]
```

LightGBM misses 192 of 995 test defaults and raises 1,584 false alarms. Logistic regression misses 392 defaults and raises 939 false alarms.

## How to run

The image serves the model stored in `models/`. It does not download the dataset and it does not train. Docker, then a prediction:

```bash
git clone <repository-url> ds-credit-risk
cd ds-credit-risk
docker build -t ds-credit-risk .
docker run --rm --name ds-credit-risk -p 8000:8000 ds-credit-risk
```

In a second shell:

```bash
curl -sS http://localhost:8000/health
curl -sS -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"limit_bal":20000,"education":2,"marriage":1,"age":24,"pay_0":2,"pay_2":2,"pay_3":-1,"pay_4":-1,"pay_5":-2,"pay_6":-2,"bill_amt1":3913,"bill_amt2":3102,"bill_amt3":689,"bill_amt4":0,"bill_amt5":0,"bill_amt6":0,"pay_amt1":0,"pay_amt2":689,"pay_amt3":0,"pay_amt4":0,"pay_amt5":0,"pay_amt6":0}'
```

Health response:

```json
{"status":"ok","model":"lightgbm","threshold":0.29}
```

Predict response for that body (the first row of the UCI file, id 1; the recorded label is default; `id` and `sex` are not accepted):

```json
{"probability_of_default":0.513262,"predicted_default":1,"threshold":0.29}
```

`predicted_default` is 1 when the probability is at least the served threshold. Stop the server with Ctrl+C.

Retrain and refresh the numbers (Python 3.11+, network access to the UCI URL). On Debian or Ubuntu, `libgomp1` has to be installed first because the LightGBM wheel loads it. The Docker image installs that library itself.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
python -m src.serve
```

`python -m src.train` writes the model and validation metrics. `python -m src.evaluate` prints the test confusion matrices and gain shares, and adds them to `models/metadata.json`. Rebuild the image after a retrain so the container serves the new file.

## Limitations

- The file is Taiwanese credit-card behavior from April–September 2005. It is not a generic personal-loan book, and the score will move if the portfolio or the year changes.
- The cost ratio 5:1 is a stated preference, not an estimate of loss given default. A different ratio moves the threshold, the F1, and the confusion matrix. ROC-AUC does not move with that ratio.
- Validation was used for early stopping, the three-configuration search, and the threshold. Quote the test row. This is one split (seed 42), so the table has no interval.
- Scores are not calibrated. The threshold was chosen on the model's raw probabilities.
- `sex` is unused. `education` and `marriage` remain in the model and can still proxy demographics.
- Early stopping kept 13 trees because validation AUC peaked there. Recent repayment status accounts for most of the gain (`pay_0` share 0.588). A book where that status is missing will not get the same AUC.
- At threshold 0.29 the test fold flags 1,584 non-defaults in order to hold missed defaults at 192.

## Decision log

1. **Dataset.** UCI Default of Credit Card Clients, 30,000 rows, CC BY 4.0, direct zip with a pinned checksum. Discarded: German Credit, because it has about 1,000 rows and this brief asks for at least 10,000. Discarded: Give Me Some Credit, because the Kaggle copy needs an account and an API token.

2. **Model family.** LightGBM, picked over the logistic baseline by validation ROC-AUC (0.7817 vs 0.7311). Discarded: XGBoost, so the image carries one boosting library. Discarded: a neural net, which this tabular file and this stack do not need.

3. **Metrics.** Selection uses validation ROC-AUC. The cutoff minimizes `5 * FN + 1 * FP` on validation, because a missed default is the expensive error. F1 and PR-AUC are reported at that cutoff and do not choose the model. Discarded: choosing the model by F1, which would follow the cutoff instead of the ranking. Discarded: SHAP, in favor of LightGBM gain, so the environment does not add another package.

4. **Imbalance.** LightGBM `scale_pos_weight` is the training-fold negative/positive ratio, 3.520. Logistic regression uses `class_weight="balanced"`. Both then use the same validation cost rule. Discarded: SMOTE or any synthetic oversampling, which would add a library and invent rows. Discarded: dropping majority rows.

5. **Columns and dirty codes.** Undocumented education and marriage codes are mapped to "other" and the rows stay. Negative bills and the payment tail stay. `sex` is not a feature. Discarded: deleting the 345 + 54 dirty-code rows. Discarded: winsorizing payments inside the served model. The trees can split around the tail, and the linear baseline uses a standard scaler.

6. **Serving.** A LightGBM text file plus one FastAPI process in Docker. The image already contains the model, so a clone can score while UCI is offline. Discarded: fitting during `docker build`, which would add the download and the fit to every image build. Discarded: Kubernetes, Spark, a full MLflow server, and a custom UI.
