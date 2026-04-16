import pandas as pd
from langchain_core.tools  import tool

_df = pd.read_csv("data/fraudTrain.csv", dtype={"cc_num": "int64"})


import math

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
    return R * 2 * math.asin(math.sqrt(a))

@tool
def transaction_inspector(trans_num: str)-> dict:
    """
    Returns facts for a single transaction: amount, datetime, merchant, category, and distance from the cardholder's home in km.

    Call this when you need to inspect one specific transaction by its trans_num — typically to compare against 
    the customer's baseline profile during fraud investigation.
    
    """
    row = _df[_df['trans_num']== trans_num].iloc[0]
    amount = float(row['amt'])
    trans_datetime = str(row['trans_date_trans_time'])
    merchant = str(row['merchant'])
    category = str(row['category'])
    distance_from_home_km = _haversine_km( row['lat'],row['long'],row['merch_lat'],row['merch_long'] )
    
    return {
    'amount': amount,
    'datetime' : trans_datetime,
    'merchant': merchant,
    'category' :category,
    "cc_num": int(row['cc_num']),
    'distance_from_home_km': distance_from_home_km }