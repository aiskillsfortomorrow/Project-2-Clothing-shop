# Clothing Shop Sales and Customer Feedback Assistant (Project 2)

A five-tab Streamlit app for a small clothing shop owner: demand estimate, product profiles,
image category suggestion, customer-comment sentiment, and a weekly review.

**All numbers are demonstration results on generated classroom data** - not a real shop forecast,
and not evidence that a discount increases profit.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free)
1. Create a GitHub Free repository and upload every file in this folder (Add file -> Upload files).
2. On Streamlit Community Cloud: New app -> pick the repo, branch `main`, file `app.py` -> Deploy.
   Use Python 3.13 in Advanced settings to match training.
3. Optional: in the app's **Secrets** panel add
   ```
   GROQ_API_KEY = "..."
   GROQ_MODEL = "openai/gpt-oss-20b"
   ```
   Without a key the app works fully in **Offline summary** mode.

## Models (trained once in Colab, loaded only)
| File | Model | Test result |
|---|---|---|
| models/supervised.joblib | Decision tree (max_depth=5) | MAE 5.45 units, RMSE 6.64, R2 0.598 (mean baseline MAE 8.74) |
| models/clusters.joblib | StandardScaler + KMeans k=3 | exploratory, no score |
| models/images.keras | Small CNN, 3 epochs | accuracy 0.973, macro F1 0.973 |
| VADER (nltk_data) | Lexicon sentiment | accuracy 0.9, macro F1 0.902 on 30 comments |

## Limits
- Sales table is generated; images are Fashion MNIST (grey 28x28); comments are authored examples.
- Image model only knows top / trouser / bag; scores are not calibrated probabilities.
- English-only sentiment; negation, sarcasm and mixed opinions can be wrong.
- Groq is optional, Free Plan only; quota errors fall back to the Offline summary.

## Data sources
See `source_notices/` (Fashion MNIST licence, NLTK/VADER licence, attribution).
Full training image arrays (`images.npz`) are deliberately **not** in this repository.

Demo link: _add your Streamlit URL here_
