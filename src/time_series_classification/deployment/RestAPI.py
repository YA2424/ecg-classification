import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

app = FastAPI()
model = mlflow.pyfunc.load_model("src/time_series_classification/deployment/best_model")


class TimeSeriesData(BaseModel):
    data: list


@app.post("/predict")
async def predict(time_series: TimeSeriesData):
    data = np.array(time_series.data).reshape(1, -1, 1).astype("float32")
    prediction = model.predict(data)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    labels = {0: "Normal", 1: "R-on-T PVC", 2: "PVC", 3: "SP or EB", 4: "UB"}
    return {"predicted_class": labels[predicted_class], "confidence": prediction[0].tolist()}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
