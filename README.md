# Churn Prediction Model

Rule-based client churn prediction scoring system for Power Digital Marketing.

## Quick Start

```python
from model import ChurnScorer

scorer = ChurnScorer()
result = scorer.score(client_df)

# Get high-risk clients
high_risk = result.get_high_risk("High")
```

## Scoring Components

- **Tier Risk** (0-25): Core > Growth > Enterprise
- **Engagement Depth** (0-20): Single blueprint = higher risk
- **Contract Stability** (0-20): 9-month pivot point
- **Renewal Urgency** (0-25): Closer to end = higher risk
- **Financial Health** (0-15): Negative retainer change = risk
- **Initial Commitment** (0-15): Short initial contract = risk

## Risk Levels

| Level | Score | Action |
|-------|-------|--------|
| Low | 0-30 | Standard engagement |
| Medium | 31-50 | Monitor |
| High | 51-70 | Intervention |
| Critical | 71+ | Escalate |

## Running Tests

```bash
uv run pytest tests/ -v
```
