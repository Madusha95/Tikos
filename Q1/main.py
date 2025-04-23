import asyncio
from datetime import datetime

import httpx
import motor.motor_asyncio
from fastapi import FastAPI, HTTPException
from bson import ObjectId

app = FastAPI()

# MongoDB setup
MONGO_URI = "mongodb+srv://eng25eastlink:CSlQQiGy9PP8vdAw@cluster0.f5ggthv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "electronics_store"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# API URLs
SOLAR_API_URL = "http://localhost:8001/solar-panels"
EV_API_URL = "http://localhost:8002/ev-chargers"

# Helper to convert ObjectId to str
def serialize_mongo_document(doc):
    doc["_id"] = str(doc["_id"])
    return doc

# Transformation logic
def transform_solar(panel):
    return {
        "model": panel["model"],
        "capacity_kw": panel["capacity_kw"],
        "efficiency_percent": panel["efficiency"],
        "manufacturer": panel["manufacturer"],
        "source": "solar",
        "ingested_at": datetime.utcnow()
    }

def transform_ev(charger):
    return {
        "location": charger["location"],
        "max_power_kw": charger["max_power_kw"],
        "plug_type": charger["plug_type"],
        "status": charger["status"],
        "source": "ev",
        "ingested_at": datetime.utcnow()
    }

# API endpoint to fetch and store data
@app.get("/ingest-data")
async def ingest_data():
    try:
        async with httpx.AsyncClient() as client_http:
            solar_resp, ev_resp = await asyncio.gather(
                client_http.get(SOLAR_API_URL),
                client_http.get(EV_API_URL)
            )

        solar_data = solar_resp.json()
        ev_data = ev_resp.json()

        transformed_solar = [transform_solar(item) for item in solar_data]
        transformed_ev = [transform_ev(item) for item in ev_data]

        await db.solar_panels.insert_many(transformed_solar)
        await db.ev_chargers.insert_many(transformed_ev)

        return {"message": "✅ Data successfully ingested and stored!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Endpoint to fetch stored data
@app.get("/solar-panels")
async def get_solar_panels():
    panels = await db.solar_panels.find().to_list(length=100)
    return [serialize_mongo_document(panel) for panel in panels]

@app.get("/ev-chargers")
async def get_ev_chargers():
    chargers = await db.ev_chargers.find().to_list(length=100)
    return [serialize_mongo_document(charger) for charger in chargers]
