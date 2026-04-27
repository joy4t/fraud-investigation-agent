"""Report generator tool - assembles structured fraud investigation report."""


import pandas as pd
from langchain_core.tools import tool
from src.tools._data import df
from src.tools.risk_scorer import risk_scorer as risk_scorer_tool
from src.tools.transaction_inspector import transaction_inspector as inspector_tool
from src.tools.customer_profiler import customer_profiler as profiler_tool



@tool
def report_generator(trans_num: str, cc_num: str) -> dict:
    """Generate a structured fraud investigation report for a single transaction.
    Assembles transaction details, customer baseline, and risk assessment
    into a five-section report that a fraud analyst can review top-to-bottom.
    Use this as the FINAL tool call after reviewing inspector and profiler output.

    Args:
        trans_num: The transaction identifier to investigate.
        cc_num: The customer's credit card number as a string.
    """
    trans_num = str(trans_num)
    cc_num = str(cc_num)

    transaction = inspector_tool.invoke({"trans_num" : trans_num})
    if "error" in transaction:
        return transaction

    customer = profiler_tool.invoke({"cc_num": cc_num})
    if "error" in customer:
        return customer

    risk = risk_scorer_tool.invoke({"trans_num": trans_num, "cc_num": cc_num})
    if "error" in risk:
        return risk

    # Determine recommendation based on risk level
    recommendations = {
        "CRITICAL": "BLOCK card immediately and escalate to senior analyst",
        "HIGH": "ESCALATE to fraud team for manual review",
        "MEDIUM": "FLAG for monitoring — review if further alerts trigger",
        "LOW": "NO ACTION — transaction appears consistent with customer history",
    }

    report = {
        "verdict": {
            "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"],
            "recommendation": recommendations[risk["risk_level"]],
        },
        "transaction_snapshot": transaction,
        "customer_baseline": customer,
        "signal_breakdown": risk["signals"],
        "raw_evidence": {
            "inspector_output": transaction,
            "profiler_output": customer,
            "risk_scorer_output": risk,
        },
    }

    return report
