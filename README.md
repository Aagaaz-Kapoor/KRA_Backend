# Meetings KRA Predictive Model API

This backend contains a predictive model for categorizing meeting descriptions and a Flask API to serve predictions to a frontend.

## Setup Instructions

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the Jupyter notebook `Meetings11.ipynb` to train the model and generate the pickle file `Meetings11.pkl`.

3. Start the Flask API server:

```
python backend/Test3.py
```

The API will be available at `http://localhost:5000`.

## API Endpoints

- `POST /predict`

  Request JSON body:

  ```json
  {
    "description": "Your meeting description text here"
  }
  ```

  Response JSON:

  ```json
  {
    "prediction": "Predicted category"
  }
  ```

- `GET /health`

  Returns API status.

## Notes

- Ensure the pickle file `Meetings11.pkl` is present in the `backend` directory before running the API.
- CORS is enabled to allow frontend applications to access the API.
