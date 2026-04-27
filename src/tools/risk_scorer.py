"""Risk scorer tool - quantifies fraud risk from transactions + customer evidence."""
import math
from langchain_core.tools import tool
from src.tools._data import df

def _amount_score(amount: float, median_amt: float, std_amt: float) -> dict:
    """Score amount deviation using z-score. Return points (0-25) and reasoning."""
    if std_amt == 0:
        z = 0.0
    else:
        z = abs(amount - median_amt) / std_amt

    if z<1:
        points = 0
    elif z < 2:
        points = 10
    elif z < 3:
        points = 20
    else: 
        points = 25
    
    return {
        "signal" : "amount_deviation",
        "points" : points,
        "details" : f"${amount:.2f} vs median ${median_amt:.2f} (z={z:.1f})"
    }

def _time_score( transaction_hour: int, typical_hour_min: int, typical_hour_max: int) -> dict:
    """Score time anamoly based on distance from typical activity windoe."""
    if typical_hour_min <= transaction_hour <= typical_hour_max:
        points = 0
        outside = 0
    else:
        outside_low = typical_hour_min - transaction_hour
        outside_high = transaction_hour - typical_hour_max
        outside = max(outside_low, outside_high)
        if outside <= 2:
            points = 10
        else:
            points = 25
    return {
        "signal" : "time_anamoly",
        "points" : points,
        "detail" : f"Hour {transaction_hour} vs typical {typical_hour_min} - {typical_hour_max} ({outside}h outside)"
    }

def _distance_score(distance_km: float) -> dict:
    """Score merchant distance from customer home."""
    if distance_km < 50:
        points = 0
    elif distance_km < 200:
        points = 10
    elif distance_km < 500:
        points = 20
    else:
        points = 25

    return {
        "signal": "distance_from_home",
        "points": points,
        "details": f"{distance_km:.1f} km from home"
    }

def _category_score(category: str, top_3_categories: list) -> dict:
    """Score category mismatch against customer's usual spending."""
    if category in top_3_categories:
        points = 0
        match = "in top 3"
    else:
        points = 15
        match = "not in top 3"
    
    return{
        "signal": "category_mismatch",
        "points": points,
        "detail": f"'{category}' {match}: {top_3_categories}"
    }

@tool
def risk_scorer(trans_num: str, cc_num: str) -> dict:
    """Compute a qualified fraud risk score (0-100) for a transaction.
    
    Combines four weighted signals: amount deivations from the customer baseline,
    transaction time vs typical activity hours, merchant distance from home, and speding
    category mismatch. Call AFTER transaction_inspector and customer_profiler to get a
    calibrated risk assessment.

    Args:
        trans_num : Transaction idetentfier to score.
        cc_num: Customer credit card number for baseline lookup.
    """
    trans_num = str(trans_num)
    cc_num = str(cc_num)

    tx_row =  df[df['trans_num'] == trans_num]
    if tx_row.empty:
        return {"error": f"Transaction {trans_num} not found"}
    tx_row = tx_row.iloc[0]

    amount = float(tx_row['amt'])
    transaction_hour = int(tx_row["trans_date_trans_time"].split(" ")[1].split(":")[0])
    category = str(tx_row["category"])
    merchant = str(tx_row["merchant"]).replace("fraud_", "")

    # Haversine for distance
    lat1, lon1 = math.radians(float(tx_row["lat"])), math.radians(float(tx_row["long"]))
    lat2, lon2 = math.radians(float(tx_row["merch_lat"])), math.radians(float(tx_row["merch_long"]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    distance_km = 6371 * 2 * math.asin(math.sqrt(a))

    # --- Look up customer baseline ---
    cust_df = df[df["cc_num"] == cc_num]
    if cust_df.empty:
        return {"error": f"Customer {cc_num} not found"}

    median_amt = float(cust_df["amt"].median())
    std_amt = float(cust_df["amt"].std())
    hours = cust_df["trans_date_trans_time"].apply(lambda x: int(x.split(" ")[1].split(":")[0]))
    q1_hour = int(hours.quantile(0.25))
    q3_hour = int(hours.quantile(0.75))
    top_3 = cust_df["category"].value_counts().head(3).index.tolist()

    # --- Score each signal ---
    signals = [
        _amount_score(amount, median_amt, std_amt),
        _time_score(transaction_hour, q1_hour, q3_hour),
        _distance_score(distance_km),
        _category_score(category, top_3),
    ]

    total = sum(s["points"] for s in signals)

    if total <= 25:
        risk_level = "LOW"
    elif total <= 50:
        risk_level = "MEDIUM"
    elif total <= 75:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        "risk_score": total,
        "risk_level": risk_level,
        "signals": signals,
        "transaction": f"{merchant} | ${amount:.2f} | {category}",
    }