# Customer Intent AI

A Streamlit application that trains a text classifier to predict customer-service intent from labeled messages.

## Features

- Upload a CSV dataset and validate its required columns
- Train a TF-IDF + Logistic Regression classification pipeline
- Review dataset rows, intent distribution, test accuracy, and a classification report
- Predict the intent of a custom customer message
- Inspect the model's top probability scores

## Dataset format

Upload a CSV file containing these columns:

| Column | Description |
|---|---|
| `utterance` | Customer message or query |
| `intent` | Correct intent label for that message |

Example:

```csv
utterance,intent
"Where is my package?",track_order
"I want to return this item",return_item
"Please cancel my subscription",cancel_subscription
```

Use a sufficiently large labeled dataset with multiple examples per intent for more reliable evaluation. The app does not include a dataset; upload your own CSV when running it.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app from `thescifi01/intent-classifier`, select branch `main` and main file path `app.py`.
4. Deploy and wait for the build to finish.

## GitHub Actions

A workflow under `.github/workflows/python-check.yml` installs dependencies and checks Python syntax on pushes and pull requests to `main`.

## Limitations

- Accuracy depends on the uploaded dataset, label quality, class balance, and train/test split.
- Model probability scores are not guaranteed real-world confidence.
- This is an educational demo, not a production-grade customer-support decision system.
