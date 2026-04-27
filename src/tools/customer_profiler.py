import pandas as pd
from langchain_core.tools  import tool
from src.tools._data import df

#df = pd.read_csv("data/fraudTrain.csv")

#df = pd.read_csv("data/fraudTrain.csv", dtype={"cc_num": "int64"})

@tool
def customer_profiler(cc_num: str) -> dict:
    """
    Build a behavioral profile for a customer given thier credit card number. 
    
    Use this tool when you need to understand a customer's normal spending patterns before judging  if a transaction is suspicious.
    Returns a baseline of normal behaviour - transaction count, typical amount, spending variablility, top merchants category, 
    typical hours and home location - to compare new transactions against.

    Args: cc_num The customer's credit card number  (integer).
    Returns: A dict with keys: total_transactions, median_amt, mean_amt, std_amt, top_3_categories, typeical_hour_range, hme_city,
    home_state, history_days. 
    
    """
    #trans_num = str(trans_num)

    cc_num = str(cc_num)
    customerdf = df[df.cc_num == cc_num].copy()

    total_transactions = customerdf.shape[0]
    median_amt = float(round(customerdf['amt'].median(), 2))
    mean_amt   = float(round(customerdf['amt'].mean(), 2))
    std_amt    = float(round(customerdf['amt'].std(), 2))
    top_3_categories = customerdf.category.value_counts().head(3).index.tolist()
    customerdf['trans_date_trans_time'] = pd.to_datetime(customerdf['trans_date_trans_time'])
    hours = customerdf['trans_date_trans_time'].dt.hour
    h25 = int(hours.quantile(0.25))
    h75 = int(hours.quantile(0.75))
    typical_hour_range =  f"{h25:02d}-{h75:02d}"
    home_state = customerdf['state'].iloc[0]
    home_city = customerdf['city'].iloc[0]
    history_days = (
    customerdf['trans_date_trans_time'].max() - customerdf['trans_date_trans_time'].min()).days

    return {
    "total_transactions": total_transactions,
    "median_amt": median_amt,
    "mean_amt": mean_amt,
    "std_amt": std_amt,
    "top_3_categories": top_3_categories,
    "typical_hour_range": typical_hour_range,
    "home_city": home_city,
    "home_state": home_state,
    "history_days": history_days,}