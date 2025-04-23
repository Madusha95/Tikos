from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import random

app = FastAPI(title="EV Charger API")

class EVCharger(BaseModel):
    id: int
    location: str
    max_power_kw: float
    plug_type: str
    status: str

@app.get("/ev-chargers", response_model=List[EVCharger])
def get_ev_chargers():
    chargers = [
        EVCharger(
            id=i,
            location=random.choice(["San Francisco", "Berlin", "Tokyo", "London", "Sydney"]),
            max_power_kw=random.choice([22, 50, 120, 350]),
            plug_type=random.choice(["Type1", "Type2", "CCS", "CHAdeMO"]),
            status=random.choice(["Available", "Occupied", "Out of Service"])
        )
        for i in range(5)
    ]
    return chargers
