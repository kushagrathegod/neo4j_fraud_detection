from fastapi import FastAPI, Header, HTTPException
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")
app = FastAPI()
# 🔐 ENV VARIABLES (for deployment)
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
API_KEY = os.getenv("API_KEY")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


# 🔧 Core function
def process_transaction(tx, data):
    query = """
    MERGE (s:Account {account_id: $sender})
    MERGE (r:Account {account_id: $receiver})

    MERGE (t:Transaction {txn_id: $txn_id})
    SET t.amount = $amount,
        t.timestamp = datetime($time),
        t.location = $location,
        t.channel = $channel

    MERGE (s)-[:SENT]->(t)
    MERGE (t)-[:TO]->(r)

    MERGE (d:Device {device_id: $device_id})
    MERGE (s)-[:USES]->(d)

    // -------- FEATURE CALCULATION --------
    WITH s

    OPTIONAL MATCH (s)-[:SENT]->(t1:Transaction)
    WITH s, count(t1) AS txn_count

    OPTIONAL MATCH (s)<-[:TO]-(t2:Transaction)
    WITH s, txn_count, count(t2) AS incoming

    OPTIONAL MATCH (s)-[:USES]->(d2:Device)
    WITH s, txn_count, incoming, count(DISTINCT d2) AS device_count

    // -------- STORE FEATURES --------
    SET s.txn_count = txn_count,
        s.incoming_count = incoming,
        s.device_count = device_count

    // -------- RISK SCORE --------
    SET s.risk_score =
        (txn_count * 0.3) +
        (incoming * 0.5) +
        (device_count * 0.2) +

        (CASE WHEN txn_count > 20 THEN 10 ELSE 0 END) +
        (CASE WHEN incoming > 15 THEN 10 ELSE 0 END) +
        (CASE WHEN device_count > 3 THEN 10 ELSE 0 END)

    RETURN s.account_id AS account,
           s.risk_score AS risk_score,
           txn_count,
           incoming,
           device_count
    """

    return tx.run(query, **data).single()


# 🚀 API Endpoint
@app.post("/check-transaction")
def check_transaction(data: dict, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    with driver.session() as session:
        result = session.execute_write(process_transaction, data)

    if result is None:
        return {"error": "Processing failed"}

    risk_score = result["risk_score"]

    return {
        "account": result["account"],
        "risk_score": round(risk_score, 2),
        "decision": "BLOCK" if risk_score > 15 else "ALLOW",
        "features": {
            "txn_count": result["txn_count"],
            "incoming": result["incoming"],
            "device_count": result["device_count"]
        }
    }


# ✅ Health check
@app.get("/")
def home():
    return {"message": "Fraud Detection API running 🚀"}