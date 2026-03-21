from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Depends
from neo4j import GraphDatabase
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
import os
import pandas as pd
from contextlib import asynccontextmanager
import xgboost as xgb
from fastapi.middleware.cors import CORSMiddleware

# ==============================
# LOAD ENV
# ==============================
load_dotenv(".env")

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
API_KEY = os.getenv("API_KEY")

if not URI or not USER or not PASSWORD:
    raise Exception("Missing Neo4j credentials")

driver = GraphDatabase.driver(
    URI,
    auth=(USER, PASSWORD),
    max_connection_pool_size=50
)

# ==============================
# LOAD ML MODEL
# Booster.load_model uses XGBoost native format (.ubj) —
# version-stable, faster than pickle, no deprecation warnings.
# ==============================
booster = xgb.Booster()
booster.load_model("fraud_model.ubj")

# ==============================
# FASTAPI LIFESPAN
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    driver.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==============================
# REQUEST MODEL
# ==============================
class Transaction(BaseModel):
    sender: str
    receiver: str
    txn_id: str
    amount: float
    time: datetime
    location: str
    channel: str
    device_id: str

# ==============================
# CORE QUERY
# ==============================
def process_transaction(tx, data):
    query = """
    MERGE (s:Account {account_id: $sender})
    ON CREATE SET s.txn_count = 0, s.incoming_count = 0

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

    // FAST COUNTERS
    SET s.txn_count = coalesce(s.txn_count, 0) + 1
    SET r.incoming_count = coalesce(r.incoming_count, 0) + 1

    WITH s

    OPTIONAL MATCH (s)-[:USES]->(d2:Device)
    WITH s, count(DISTINCT d2) AS device_count

    SET s.device_count = device_count

    // VELOCITY — exclude current transaction so count reflects prior activity only
    OPTIONAL MATCH (s)-[:SENT]->(recent:Transaction)
    WHERE recent.timestamp > datetime() - duration('PT10M')
      AND recent.txn_id <> $txn_id
    WITH s, device_count, count(recent) AS last_10min_txn

    // CHAIN
    OPTIONAL MATCH (s)-[:SENT]->(:Transaction)-[:TO]->(mid:Account)
    WITH s, device_count, last_10min_txn,
         count(DISTINCT mid) AS chain_count

    // GRAPH FLOW
    OPTIONAL MATCH path =
        (s)-[:SENT]->(t1:Transaction)-[:TO]->(a2:Account)
        -[:SENT]->(t2:Transaction)-[:TO]->(a3:Account)

    WITH s, device_count, last_10min_txn, chain_count,
         collect(DISTINCT s) + collect(DISTINCT a2) + collect(DISTINCT a3) AS nodes,
         collect(DISTINCT {
             source: s.account_id,
             target: a2.account_id,
             amount: t1.amount
         }) +
         collect(DISTINCT {
             source: a2.account_id,
             target: a3.account_id,
             amount: t2.amount
         }) AS edges

    RETURN s.account_id AS account,
           s.txn_count AS txn_count,
           coalesce(s.incoming_count, 0) AS incoming,
           device_count,
           last_10min_txn,
           chain_count,
           nodes,
           edges
    """
    return tx.run(query, **data).single()

# ==============================
# AUTH DEPENDENCY
# ==============================
def require_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

# ==============================
# MAIN API
# ==============================
@app.post("/check-transaction", dependencies=[Depends(require_api_key)])
def check_transaction(data: Transaction):
    payload = data.model_dump()
    payload["time"] = data.time.isoformat()

    with driver.session() as session:
        result = session.execute_write(process_transaction, payload)

    if result is None:
        return {"error": "Processing failed"}

    # ML FEATURES
    feature_df = pd.DataFrame([{
        "txn_count": result["txn_count"],
        "incoming": result["incoming"],
        "device_count": result["device_count"],
        "last_10min_txn": result["last_10min_txn"],
        "chain_count": result["chain_count"]
    }])

    # Booster.predict returns probabilities directly for binary classification —
    # no predict_proba wrapper needed, no sklearn fit state required.
    dmatrix = xgb.DMatrix(feature_df)
    ml_prob = float(booster.predict(dmatrix)[0])

    # PATTERNS
    # HUB_RELAY: sender both sends and receives heavily — indicates a layering/relay node,
    # not smurfing. Smurfing would require checking the receiver's incoming small-txn count.
    patterns = []
    if result["txn_count"] > 15 and result["incoming"] > 10:
        patterns.append("HUB_RELAY")

    if result["chain_count"] > 30:
        patterns.append("CHAIN_LAUNDERING")

    if result["last_10min_txn"] > 5:
        patterns.append("HIGH_VELOCITY")

    # FINAL SCORE
    rule_score = (
        (result["txn_count"] * 0.3) +
        (result["incoming"] * 0.5) +
        (result["device_count"] * 0.2)
    )

    normalized_rule = min(rule_score / 50, 1)
    final_score = (0.6 * ml_prob) + (0.4 * normalized_rule)

    if final_score > 0.75:
        decision = "BLOCK"
    elif final_score > 0.5:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    # GRAPH FORMAT
    nodes = []
    seen = set()

    for n in result["nodes"]:
        if n is None:
            continue
        acc = n["account_id"]
        if acc not in seen:
            nodes.append({"id": acc, "label": acc})
            seen.add(acc)

    edges = [
        e for e in result["edges"]
        if e.get("source") and e.get("target")
    ]

    return {
        "account": result["account"],
        "decision": decision,
        "confidence": round(ml_prob, 3),
        "final_score": round(final_score, 3),
        "patterns_detected": patterns,
        "graph": {
            "nodes": nodes,
            "edges": edges
        }
    }

# ==============================
# DATASET INGESTION (OPTIMIZED)
# ==============================
@app.post("/analyze-dataset", dependencies=[Depends(require_api_key)])
async def analyze_dataset(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)
    data = df.to_dict("records")

    query = """
    UNWIND $rows AS row

    MERGE (s:Account {account_id: row.sender})
    ON CREATE SET s.txn_count = 0, s.incoming_count = 0
    SET s.txn_count = coalesce(s.txn_count, 0) + 1

    MERGE (r:Account {account_id: row.receiver})
    ON CREATE SET r.incoming_count = 0
    SET r.incoming_count = coalesce(r.incoming_count, 0) + 1

    MERGE (t:Transaction {txn_id: row.txn_id})
    SET t.amount = row.amount,
        t.timestamp = datetime(row.time),
        t.location = row.location,
        t.channel = row.channel

    MERGE (s)-[:SENT]->(t)
    MERGE (t)-[:TO]->(r)

    MERGE (d:Device {device_id: row.device_id})
    MERGE (s)-[:USES]->(d)
    """

    with driver.session() as session:
        # .consume() forces the result to complete and surfaces any ingest errors
        session.run(query, rows=data).consume()

    return {"message": "Dataset ingested (optimized 🚀)"}

# ==============================
# FRAUD NETWORK
# ==============================
@app.get("/detect-fraud-network", dependencies=[Depends(require_api_key)])
def detect_fraud_network():

    # Only match direct transactions BETWEEN suspicious accounts —
    # not their entire neighbourhood. Prevents full-graph dumps.
    query = """
    MATCH (s:Account)
    WHERE s.txn_count > 20 OR s.incoming_count > 15
    WITH collect(s) AS suspects

    UNWIND suspects AS s
    UNWIND suspects AS r
    MATCH (s)-[:SENT]->(:Transaction)-[:TO]->(r)
    WHERE s <> r

    WITH collect(DISTINCT s) + collect(DISTINCT r) AS nodes,
         collect(DISTINCT {
             source: s.account_id,
             target: r.account_id
         }) AS edges

    RETURN nodes, edges
    """

    with driver.session() as session:
        result = session.run(query).single()

    if result is None or not result["nodes"]:
        return {
            "message": "No fraud network found",
            "graph": {"nodes": [], "edges": []}
        }

    nodes = []
    seen = set()

    for n in result["nodes"]:
        if n is None:
            continue
        acc = n["account_id"]
        if acc not in seen:
            nodes.append({"id": acc, "label": acc})
            seen.add(acc)

    edges = [
        e for e in result["edges"]
        if e.get("source") and e.get("target")
    ]

    return {
        "message": "Fraud network detected",
        "graph": {
            "nodes": nodes,
            "edges": edges
        }
    }

# ==============================
# PATTERN DETECTION
# ==============================
@app.get("/detect-pattern/{pattern_type}", dependencies=[Depends(require_api_key)])
def detect_pattern(pattern_type: str):

    if pattern_type == "circular":
        # Money loops back to the originating account via 2 hops
        query = """
        MATCH path = (a:Account)-[:SENT]->(:Transaction)-[:TO]->(b:Account)
                     -[:SENT]->(:Transaction)-[:TO]->(c:Account)
                     -[:SENT]->(:Transaction)-[:TO]->(a)
        RETURN path LIMIT 5
        """

    elif pattern_type == "chain":
        # Linear A → B → C → D transaction chain
        query = """
        MATCH path = (a:Account)-[:SENT]->(:Transaction)-[:TO]->(b:Account)
                     -[:SENT]->(:Transaction)-[:TO]->(c:Account)
                     -[:SENT]->(:Transaction)-[:TO]->(d:Account)
        RETURN path LIMIT 5
        """

    elif pattern_type == "velocity":
        # Accounts sending more than 5 transactions in the last 10 minutes
        query = """
        MATCH (s:Account)-[:SENT]->(t:Transaction)
        WHERE t.timestamp > datetime() - duration('PT10M')
        WITH s, count(t) AS txn_count
        WHERE txn_count > 5
        RETURN s.account_id AS account, txn_count
        ORDER BY txn_count DESC
        """

    elif pattern_type == "smurfing":
        # Receivers accumulating many small transactions below $10,000 —
        # classic structuring to avoid reporting thresholds
        query = """
        MATCH (r:Account)<-[:TO]-(t:Transaction)
        WHERE t.amount < 10000
        WITH r, count(t) AS small_txn_count, collect(t.amount) AS amounts
        WHERE small_txn_count > 10
        RETURN r.account_id AS account,
               small_txn_count,
               round(reduce(s=0.0, a IN amounts | s + a) / small_txn_count) AS avg_amount
        ORDER BY small_txn_count DESC
        """

    elif pattern_type == "device_sharing":
        # Single device used by more than 2 distinct accounts — mule network signal
        query = """
        MATCH (d:Device)<-[:USES]-(a:Account)
        WITH d, collect(DISTINCT a.account_id) AS accounts
        WHERE size(accounts) > 2
        RETURN d.device_id AS device,
               accounts,
               size(accounts) AS account_count
        ORDER BY account_count DESC
        """

    elif pattern_type == "rapid_movement":
        # Account receives funds then sends them out within 10 minutes —
        # classic money mule behaviour
        query = """
        MATCH (a:Account)<-[:TO]-(t_in:Transaction)
        MATCH (a)-[:SENT]->(t_out:Transaction)
        WHERE t_out.timestamp > t_in.timestamp
          AND t_out.timestamp < t_in.timestamp + duration('PT10M')
        WITH a, count(*) AS rapid_count
        WHERE rapid_count > 3
        RETURN a.account_id AS account, rapid_count
        ORDER BY rapid_count DESC
        """

    elif pattern_type == "round_tripping":
        # Money leaves an account and returns to it within 3 hops
        # with at least 80% of the original amount intact
        query = """
        MATCH (a:Account)-[:SENT]->(t1:Transaction)-[:TO]->(b:Account)
              -[:SENT]->(t2:Transaction)-[:TO]->(c:Account)
              -[:SENT]->(t3:Transaction)-[:TO]->(a)
        WHERE t2.timestamp > t1.timestamp
          AND t3.timestamp > t2.timestamp
          AND t3.amount >= t1.amount * 0.8
        RETURN DISTINCT a.account_id AS account,
               t1.amount AS sent,
               t3.amount AS returned,
               b.account_id AS hop1,
               c.account_id AS hop2
        LIMIT 10
        """

    else:
        return {
            "error": "Invalid pattern type",
            "valid_types": [
                "circular",
                "chain",
                "velocity",
                "smurfing",
                "device_sharing",
                "rapid_movement",
                "round_tripping"
            ]
        }

    with driver.session() as session:
        result = [r.data() for r in session.run(query)]

    return {
        "pattern": pattern_type,
        "results": result
    }

# ==============================
# GEOGRAPHIC ANOMALY DETECTION
# ==============================
@app.get("/detect-geo-anomaly", dependencies=[Depends(require_api_key)])
def detect_geo_anomaly():
    """
    Finds accounts transacting from locations inconsistent with
    their historical location pattern.
    """
    query = """
    MATCH (s:Account)-[:SENT]->(t:Transaction)
    WITH s, t.location AS loc, count(*) AS freq
    ORDER BY s.account_id, freq DESC

    WITH s, collect(loc)[0] AS usual_location

    MATCH (s)-[:SENT]->(t2:Transaction)
    WHERE t2.location <> usual_location
    WITH s, usual_location,
         collect(DISTINCT t2.location) AS anomalous_locations,
         count(t2) AS anomaly_count
    WHERE anomaly_count > 0

    RETURN s.account_id AS account,
           usual_location,
           anomalous_locations,
           anomaly_count
    ORDER BY anomaly_count DESC
    """

    with driver.session() as session:
        result = [r.data() for r in session.run(query)]

    return {
        "pattern": "geographic_anomaly",
        "results": result
    }

# ==============================
# HEALTH
# ==============================
@app.get("/")
def home():
    return {"message": "Fraud Detection API running 🚀"}
