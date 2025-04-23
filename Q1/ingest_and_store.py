import asyncio
import httpx
import motor.motor_asyncio  # Async MongoDB client
from datetime import datetime


MONGO_URI = "mongodb+srv://eng25eastlink:VxPIquCb43pPjGzg@cluster0.uehjnvb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "electronics_store"

SOLAR_API_URL = "http://localhost:8001/solar-panels"
EV_API_URL = "http://localhost:8002/ev-chargers"

# Initialize MongoDB
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

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

# Async data fetching + storing
async def fetch_and_store():
    async with httpx.AsyncClient() as client:
        solar_resp, ev_resp = await asyncio.gather(
            client.get(SOLAR_API_URL),
            client.get(EV_API_URL)
        )

        solar_data = solar_resp.json()
        ev_data = ev_resp.json()

        # Transform data
        transformed_solar = [transform_solar(item) for item in solar_data]
        transformed_ev = [transform_ev(item) for item in ev_data]

        #print("Transformed Solar Data:", transformed_solar)
        #print("Transformed EV Data:", transformed_ev)

        # Insert into MongoDB
        await db.solar_panels.insert_many(transformed_solar)
        await db.ev_chargers.insert_many(transformed_ev)

        print("✅ Data successfully ingested and stored in MongoDB!")

# Entry point
if __name__ == "__main__":
    asyncio.run(fetch_and_store())
