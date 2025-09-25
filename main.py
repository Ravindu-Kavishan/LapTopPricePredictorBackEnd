from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define your input model
class LaptopData(BaseModel):
    company: str
    typename: str
    cpu_brand: str
    gpu_brand: str
    inches: float
    weight: float
    touchscreen: bool
    ips: bool
    fullhd: bool
    os: str
    ram: str

# Columns used in your trained model
MODEL_COLUMNS = [
    'inches', 'ram', 'weight', 'touchscreen', 'ips', 'fullhd',
    'company_Acer', 'company_Apple', 'company_Asus', 'company_Dell',
    'company_HP', 'company_Lenovo', 'company_MSI', 'company_Other',
    'company_Toshiba', 'typename_2 in 1 Convertible', 'typename_Gaming',
    'typename_Netbook', 'typename_Notebook', 'typename_Ultrabook',
    'typename_Workstation', 'opsys_Android', 'opsys_Chrome OS',
    'opsys_Linux', 'opsys_Mac OS X', 'opsys_No OS', 'opsys_Windows 10',
    'opsys_Windows 10 S', 'opsys_Windows 7', 'opsys_macOS',
    'CPU_Brand_AMD Ryzen', 'CPU_Brand_Intel i3', 'CPU_Brand_Intel i5',
    'CPU_Brand_Intel i7', 'CPU_Brand_Low-end Intel', 'CPU_Brand_Other AMD',
    'GPU_Brand_AMD', 'GPU_Brand_Intel', 'GPU_Brand_Nvidia GeForce',
    'GPU_Brand_Nvidia Quadro', 'OS_Android', 'OS_Chrome OS', 'OS_Linux',
    'OS_No OS', 'OS_Windows 10', 'OS_Windows 7', 'OS_macOS'
]

@app.post("/predict")
async def predict_laptop(data: LaptopData):
    try:
        loaded_model = joblib.load('random_forest_laptop_model.pkl')

        # Create a zero-filled DataFrame with 1 row
        Xvalue = pd.DataFrame(np.zeros((1, len(MODEL_COLUMNS))), columns=MODEL_COLUMNS)

        # Fill in numeric and boolean values
        Xvalue.at[0, 'inches'] = data.inches
        Xvalue.at[0, 'weight'] = data.weight
        Xvalue.at[0, 'ram'] = int(data.ram)
        Xvalue.at[0, 'touchscreen'] = int(data.touchscreen)
        Xvalue.at[0, 'ips'] = int(data.ips)
        Xvalue.at[0, 'fullhd'] = int(data.fullhd)

        # Fill in one-hot encoded categorical values if column exists
        if f"company_{data.company}" in MODEL_COLUMNS:
            Xvalue.at[0, f"company_{data.company}"] = 1
        else:
            Xvalue.at[0, "company_Other"] = 1

        if f"typename_{data.typename}" in MODEL_COLUMNS:
            Xvalue.at[0, f"typename_{data.typename}"] = 1

        if f"CPU_Brand_{data.cpu_brand}" in MODEL_COLUMNS:
            Xvalue.at[0, f"CPU_Brand_{data.cpu_brand}"] = 1
        else:
            Xvalue.at[0, "CPU_Brand_Other AMD"] = 1

        if f"GPU_Brand_{data.gpu_brand}" in MODEL_COLUMNS:
            Xvalue.at[0, f"GPU_Brand_{data.gpu_brand}"] = 1

        if f"OS_{data.os}" in MODEL_COLUMNS:
            Xvalue.at[0, f"OS_{data.os}"] = 1

        # Predict
        prediction = loaded_model.predict(Xvalue.values)


        return {
            "prediction": float(prediction[0]),
            "message": "Prediction successful"
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "error": str(e),
            "message": "Prediction failed"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)