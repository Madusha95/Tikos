from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import random

app = FastAPI(title="Solar Panel API")

class SolarPanel(BaseModel):
    id: int
    model: str
    capacity_kw: float
    efficiency: float
    manufacturer: str

@app.get("/solar-panels", response_model=List[SolarPanel])
def get_solar_panels():
    panels = [
        SolarPanel(
            id=i,
            model=f"SP-{1000+i}",
            capacity_kw=round(random.uniform(3.0, 10.0), 2),
            efficiency=round(random.uniform(15.0, 22.0), 2),
            manufacturer=random.choice(["SunPower", "LG", "Trina", "JA Solar"])
        )
        for i in range(5)
    ]
    return panels
