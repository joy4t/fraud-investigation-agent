"""Shared data loader — single CSV load for all tools."""

import pandas as pd
import os

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fraudTrain.csv")
df = pd.read_csv(_DATA_PATH) #, dtype={"cc_num": str})
df["cc_num"] = df["cc_num"].astype(int).astype(str)