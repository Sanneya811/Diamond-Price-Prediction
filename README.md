# Diamond Price Prediction

Streamlit app that predicts diamond price from cut, color, clarity, carat, depth, table, x, y, and z.

## Run locally

```bash
cd "c:\machine learning\machine learning\environment2\Diamond Prediction"
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on GitHub and Streamlit Community Cloud

1. Create a GitHub repository.
2. Push this folder to the repo.
3. On Streamlit Community Cloud, connect the GitHub repo.
4. Set the main file path to `app.py`.
5. Deploy.

## Files

- `app.py` - Streamlit app entry point
- `diamonds.csv` - dataset used to train the model
- `requirements.txt` - Python dependencies