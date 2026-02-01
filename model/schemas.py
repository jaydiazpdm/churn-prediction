"""
Data schema definitions for churn prediction model.

Uses Pandera for runtime validation of input DataFrames to ensure
data quality and catch pipeline errors before scoring.
"""

import pandera as pa
from pandera import Column, Check, DataFrameSchema


# Schema for scoring input data
SCORING_INPUT_SCHEMA = DataFrameSchema(
    {
        "CLIENT_ID": Column(
            str,
            nullable=False,
            unique=True,
            description="Unique client identifier"
        ),
        "TIER_NAME": Column(
            str,
            nullable=False,
            checks=Check.isin(["Core", "Growth", "Enterprise"]),
            description="Client tier (Core, Growth, or Enterprise)"
        ),
        "TOTAL_BLUEPRINTS": Column(
            int,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(100),  # Reasonable upper bound
            ],
            description="Total number of blueprints for the client"
        ),
        "CONTRACT_DURATION": Column(
            int,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(120),  # Max 10 years in months
            ],
            description="Contract duration in months"
        ),
        "MONTHS_UNTIL_END": Column(
            int,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(-12),  # Allow some overdue
                Check.less_than_or_equal_to(120),
            ],
            description="Months until contract ends (negative if overdue)"
        ),
        "FIRST_LENGTH": Column(
            int,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(120),
            ],
            description="Length of first contract in months"
        ),
        "DIFF_RETAINER": Column(
            float,
            nullable=True,  # Allow null for this field
            checks=[
                Check.greater_than_or_equal_to(-1.0),  # -100% change (lost all revenue)
                Check.less_than_or_equal_to(5.0),      # +500% change (5x growth)
            ],
            description="Relative change in retainer (e.g., -0.5 = -50% decrease)"
        ),
    },
    strict=False,  # Allow extra columns (for flexibility with additional fields)
    coerce=True,   # Try to coerce types automatically
    description="Schema for churn prediction scoring input data"
)


# Schema for scoring output data
SCORING_OUTPUT_SCHEMA = DataFrameSchema(
    {
        "CLIENT_ID": Column(str, nullable=False),
        "RISK_SCORE": Column(
            int,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(200),  # Flexible upper bound
            ]
        ),
        "RISK_LEVEL": Column(
            str,
            nullable=False,
            checks=Check.isin(["Low", "Medium", "High", "Critical"])
        ),
    },
    strict=False,  # Allow component columns
    description="Schema for churn prediction scoring output data"
)
