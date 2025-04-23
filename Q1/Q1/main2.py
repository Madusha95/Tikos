import asyncio
from datetime import datetime
import logging

import httpx
import motor.motor_asyncio
from fastapi import FastAPI, HTTPException
from bson import ObjectId

app = FastAPI()

# Setup Logging to File
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
log_file = "app.log"

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

logger = logging.getLogger("IngestionService")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# MongoDB Setup
MONGO_URI = "mongodb+srv://eng25eastlink:CSlQQiGy9PP8vdAw@cluster0.f5ggthv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "electronics_store"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# External API URLs
SOLAR_API_URL = "http://localhost:8001/solar-panels"
EV_API_URL = "http://localhost:8002/ev-chargers"

# Utility: Convert Mongo ObjectId to string
def serialize_mongo_document(doc):
    doc["_id"] = str(doc["_id"])
    return doc

# Data Transformers
def transform_solar(panel: dict) -> dict:
    return {
        "model": panel.get("model"),
        "capacity_kw": panel.get("capacity_kw"),
        "efficiency_percent": panel.get("efficiency"),
        "manufacturer": panel.get("manufacturer"),
        "source": "solar",
        "ingested_at": datetime.utcnow()
    }

def transform_ev(charger: dict) -> dict:
    return {
        "location": charger.get("location"),
        "max_power_kw": charger.get("max_power_kw"),
        "plug_type": charger.get("plug_type"),
        "status": charger.get("status"),
        "source": "ev",
        "ingested_at": datetime.utcnow()
    }

# Endpoint: Ingest Data from External APIs
@app.get("/ingest-data")
async def ingest_data():
    try:
        logger.info(" Starting data ingestion process...")

        async with httpx.AsyncClient() as client_http:
            solar_task = client_http.get(SOLAR_API_URL)
            ev_task = client_http.get(EV_API_URL)
            solar_resp, ev_resp = await asyncio.gather(solar_task, ev_task)

        if solar_resp.status_code != 200:
            msg = f"Solar API error: {solar_resp.status_code}"
            logger.error(msg)
            raise HTTPException(status_code=solar_resp.status_code, detail=msg)

        if ev_resp.status_code != 200:
            msg = f"EV API error: {ev_resp.status_code}"
            logger.error(msg)
            raise HTTPException(status_code=ev_resp.status_code, detail=msg)

        solar_data = solar_resp.json()
        ev_data = ev_resp.json()

        logger.info(f" Retrieved {len(solar_data)} solar panels")
        logger.info(f" Retrieved {len(ev_data)} EV chargers")

        transformed_solar = [transform_solar(item) for item in solar_data]
        transformed_ev = [transform_ev(item) for item in ev_data]

        if transformed_solar:
            await db.solar_panels.insert_many(transformed_solar)
            logger.info(" Solar panel data inserted into DB")

        if transformed_ev:
            await db.ev_chargers.insert_many(transformed_ev)
            logger.info("EV charger data inserted into DB")

        return {"message": " Data successfully ingested and stored!"}

    except Exception as e:
        logger.exception("Ingestion failed due to an unexpected error.")
        raise HTTPException(status_code=500, detail="Ingestion failed due to an internal error.")

# Endpoint: Get Stored Solar Panels
@app.get("/solar-panels")
async def get_solar_panels():
    try:
        panels = await db.solar_panels.find().to_list(length=100)
        return [serialize_mongo_document(panel) for panel in panels]
    except Exception as e:
        logger.exception("Failed to fetch solar panels")
        raise HTTPException(status_code=500, detail="Failed to fetch solar panels")

# Endpoint: Get Stored EV Chargers
@app.get("/ev-chargers")
async def get_ev_chargers():
    try:
        chargers = await db.ev_chargers.find().to_list(length=100)
        return [serialize_mongo_document(charger) for charger in chargers]
    except Exception as e:
        logger.exception("Failed to fetch EV chargers")
        raise HTTPException(status_code=500, detail="Failed to fetch EV chargers")
