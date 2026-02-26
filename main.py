from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Load model and accuracy
project_dir = Path(__file__).resolve().parent
with (project_dir / "real_model.pkl").open("rb") as f:
    model, accuracy, features = pickle.load(f)

# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=str(project_dir / "templates" / "static")), name="static")

templates = Jinja2Templates(directory=str(project_dir / "templates"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "accuracy": round(accuracy * 100, 2)
    })

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            GrLivArea: Optional[float] = Form(None),
            BedroomAbvGr: Optional[int] = Form(None),
            FullBath: Optional[int] = Form(None),
            LotArea: Optional[float] = Form(None),
            YearBuilt: Optional[int] = Form(None),
            area: Optional[float] = Form(None),
            bedrooms: Optional[int] = Form(None),
            age: Optional[int] = Form(None),
            location_score: Optional[int] = Form(None)):

    try:
        # Accept both model-native feature names and your current form field names.
        if GrLivArea is None:
            GrLivArea = area
        if BedroomAbvGr is None:
            BedroomAbvGr = bedrooms
        if FullBath is None:
            FullBath = location_score
        if LotArea is None:
            LotArea = area
        if YearBuilt is None and age is not None:
            YearBuilt = date.today().year - age

        if None in (GrLivArea, BedroomAbvGr, FullBath, LotArea, YearBuilt):
            raise ValueError("Missing required values for prediction.")

        input_data = pd.DataFrame([{
            "GrLivArea": GrLivArea,
            "BedroomAbvGr": BedroomAbvGr,
            "FullBath": FullBath,
            "LotArea": LotArea,
            "YearBuilt": YearBuilt
        }], columns=features)

        prediction = model.predict(input_data)
        result = float(prediction[0])

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": result,
            "accuracy": round(accuracy * 100, 2)
        })

    except Exception:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Invalid input values",
            "accuracy": round(accuracy * 100, 2)
        })
