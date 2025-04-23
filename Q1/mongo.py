import motor.motor_asyncio

client = motor.motor_asyncio.AsyncIOMotorClient(
    "mongodb+srv://eng25eastlink:CSlQQiGy9PP8vdAw@cluster0.f5ggthv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
db = client["electronics_store"]

async def test_connection():
    try:
        collections = await db.list_collection_names()
        print("✅ Connection OK! Collections:", collections)
    except Exception as e:
        print("❌ Connection failed:", e)

import asyncio
asyncio.run(test_connection())
