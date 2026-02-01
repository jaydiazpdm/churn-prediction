# Churn Prediction - ML Testing Guide

**Owner:** Laura (Junior DS)
**Last Updated:** 2026-02-01
**Status:** Phase 1 (P0) Complete ✅

---

## 🎯 Quick Start

```bash
# Install test dependencies
python3 -m pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run only critical P0 tests
pytest tests/test_ml_quality.py::TestModelPerformance -v
pytest tests/test_data_quality.py -v
pytest tests/test_production.py -v

# Generate coverage report
pytest tests/ --cov=model --cov=experiments --cov-report=html
open htmlcov/index.html
```

---

## 📊 Test Coverage

### Current Status
- **Total tests:** 81 (30 existing + 51 new)
- **All tests passing:** ✅ 30/30 existing tests
- **Phase completed:** P0 Critical Blockers
- **Coverage target:** >80%

### Test Categories

| Category | File | Tests | Status |
|----------|------|-------|--------|
| **Component Tests** | `test_components.py` | 12 | ✅ Passing |
| **Scorer Tests** | `test_scorer.py` | 18 | ✅ Passing |
| **ML Quality** | `test_ml_quality.py` | 15 | ✅ Implemented |
| **Data Quality** | `test_data_quality.py` | 12 | ✅ Implemented |
| **Production** | `test_production.py` | 12 | ✅ Implemented |
| **Regression** | `test_regression.py` | 9 | ✅ Implemented |
| **Integration** | `test_integration.py` | 11 | ✅ Implemented |

---

## 🚀 Test Philosophy

### Our Priorities
1. **Kill Criteria Enforcement:** Model must maintain >50% accuracy
2. **Recall First:** Must catch 70%+ of churners (false positives acceptable)
3. **Production Ready:** Fast (<5s for 10K clients), reliable, clear errors
4. **Fairness:** Consistent performance across client tiers
5. **No Regressions:** Performance never degrades >5%

### Key Metrics
- **Accuracy:** >50% (kill criteria), target 65%+
- **Recall:** >70% (critical - catch churners)
- **F1 Score:** >0.40 baseline, target 0.55+
- **Performance:** 10K clients in <5 seconds
- **Memory:** <500MB for 10K clients

---

## 📁 Test Organization

```
tests/
├── conftest.py                    # Fixtures for all tests
├── fixtures/
│   └── baseline_metrics.json      # Performance baselines
│
├── test_components.py             # Unit tests (12 tests)
├── test_scorer.py                 # Integration tests (18 tests)
│
├── test_ml_quality.py             # ML-specific tests (15 tests)
│   ├── TestModelPerformance       # Kill criteria, recall, F1
│   ├── TestThresholdOptimization  # Threshold stability
│   ├── TestScoreDistribution      # Score separation quality
│   └── TestModelFairness          # Tier parity (FPR, recall)
│
├── test_data_quality.py           # Data validation (12 tests)
│   ├── TestDataQuality            # Schema validation
│   ├── TestDataRanges             # Range checks, outliers
│   └── TestDataConsistency        # Duplicates, logic
│
├── test_production.py             # Production readiness (12 tests)
│   ├── TestProductionPerformance  # Latency, memory
│   ├── TestErrorHandling          # Clear errors
│   ├── TestProductionMonitoring   # Volume, proportions
│   └── TestLogging                # Logs and artifacts
│
├── test_regression.py             # Performance baselines (9 tests)
│   ├── TestPerformanceRegression  # No degradation
│   ├── TestComponentDeterminism   # Same input = same output
│   └── TestConfigBackwardCompatibility  # Old configs work
│
└── test_integration.py            # End-to-end (11 tests)
    ├── TestEndToEndPipeline       # Full workflows
    ├── TestExperimentConfigs      # Config validation
    └── TestDataPipeline           # Split integrity
```

---

## 🧪 Running Tests

### By Category
```bash
# ML quality tests
pytest tests/test_ml_quality.py -v

# Data quality tests
pytest tests/test_data_quality.py -v

# Production readiness
pytest tests/test_production.py -v

# Regression tests
pytest tests/test_regression.py -v

# Integration tests
pytest tests/test_integration.py -v
```

### By Priority
```bash
# P0 Critical (pre-deployment gate)
pytest tests/test_ml_quality.py::TestModelPerformance -v
pytest tests/test_data_quality.py::TestDataQuality -v
pytest tests/test_production.py::TestProductionPerformance -v
pytest tests/test_integration.py::TestEndToEndPipeline -v

# P1 Sustainable operations (fairness, monitoring)
pytest tests/test_ml_quality.py::TestModelFairness -v
pytest tests/test_production.py::TestProductionMonitoring -v
```

### Specific Tests
```bash
# Single test
pytest tests/test_ml_quality.py::TestModelPerformance::test_accuracy_above_kill_criteria -v

# Pattern matching
pytest tests/ -k "accuracy" -v
pytest tests/ -k "fairness" -v

# Slow tests (marked with @pytest.mark.slow)
pytest tests/ --runslow -v
```

### With Coverage
```bash
# Terminal report
pytest tests/ --cov=model --cov=experiments --cov-report=term-missing

# HTML report (better visualization)
pytest tests/ --cov=model --cov=experiments --cov-report=html
open htmlcov/index.html
```

### Performance Analysis
```bash
# Show 10 slowest tests
pytest tests/ --durations=10

# Show all durations
pytest tests/ --durations=0
```

---

## 🔍 Understanding Test Failures

### Common Failure Patterns

#### 1. Performance Degradation
```
FAILED test_ml_quality.py::test_accuracy_no_regression
AssertionError: Accuracy degraded: 65.2% -> 62.1% (degradation: 3.1%)
```

**Fix:**
- Review recent code changes
- Check if data pipeline changed
- Verify experiment config is correct
- If improvement is real, update baseline in `tests/fixtures/baseline_metrics.json`

#### 2. Schema Validation Error
```
FAILED test_data_quality.py::test_scorer_validates_schema
pandera.errors.SchemaError: Column 'TIER_NAME' has invalid values: ['InvalidTier']
```

**Fix:**
- Check data source for data quality issues
- Verify tier names match ["Core", "Growth", "Enterprise"]
- Update schema in `model/schemas.py` if business requirements changed

#### 3. Performance Too Slow
```
FAILED test_production.py::test_batch_scoring_performance_10k_clients
AssertionError: Too slow: 7.23s for 10K clients (target: <5s)
```

**Fix:**
- Profile code with `pytest --profile`
- Check for non-vectorized operations (row-by-row loops)
- Verify data is preprocessed correctly
- Consider optimizing component scorers

#### 4. Fairness Test Failure
```
FAILED test_ml_quality.py::test_recall_parity_across_tiers
AssertionError: Recall disparity across tiers: {'Core': 0.75, 'Growth': 0.68, 'Enterprise': 0.52}
```

**Fix:**
- Review scoring weights for tier bias
- Check if one tier has unusual data distribution
- Consider if business logic justifies difference
- Adjust thresholds if disparity is acceptable

---

## 🛠️ Debugging Tests

### Step 1: Run with Verbose Output
```bash
pytest tests/test_ml_quality.py -v -s
# -v: verbose (show test names)
# -s: show print statements
```

### Step 2: Run Single Failing Test
```bash
pytest tests/test_ml_quality.py::TestModelPerformance::test_accuracy_above_kill_criteria -v
```

### Step 3: Add Breakpoint
```python
def test_accuracy_above_kill_criteria(self):
    runner = ExperimentRunner()
    result = runner.run(config)

    import pdb; pdb.set_trace()  # Pause here
    # or
    breakpoint()  # Python 3.7+

    assert result.metrics["test_accuracy"] >= 0.50
```

### Step 4: Check Experiment Logs
```bash
# View recent experiment logs
ls -lt experiments/logs/ | head -5

# Check specific experiment
cat experiments/logs/exp_20260201_0001.json | jq .
```

### Step 5: Review Test Data
```bash
# Verify data files exist
ls -lh experiments/data/

# Check data quality
python3 -c "
import pandas as pd
df = pd.read_csv('experiments/data/test.csv')
print(df.info())
print(df.describe())
"
```

---

## 📝 Writing New Tests

### Test Template
```python
class TestNewFeature:
    """Tests for [feature description]."""

    @pytest.fixture
    def sample_data(self):
        """Create test data."""
        return pd.DataFrame({
            "CLIENT_ID": ["TEST1", "TEST2"],
            "TIER_NAME": ["Core", "Growth"],
            # ... other required columns
        })

    def test_feature_works_correctly(self, sample_data):
        """Feature should [expected behavior]."""
        # Arrange
        scorer = ChurnScorer()

        # Act
        result = scorer.score(sample_data)

        # Assert
        assert len(result.df) == 2, "Should score all clients"
        assert result.df["RISK_SCORE"].min() >= 0, "Scores should be non-negative"
```

### Best Practices

1. **Clear Test Names**
   - ✅ `test_accuracy_above_kill_criteria`
   - ❌ `test_accuracy`

2. **Descriptive Docstrings**
   ```python
   def test_recall_meets_constraint(self):
       """Model must maintain 70%+ recall for catching churners."""
   ```

3. **Assert Messages**
   ```python
   assert accuracy >= 0.50, \
       f"Model failed kill criteria: {accuracy:.1%} < 50%"
   ```

4. **Use Fixtures**
   ```python
   @pytest.fixture
   def baseline_config(self):
       return ExperimentConfig.from_yaml("configs/baseline.yaml")
   ```

5. **Parametrize for Multiple Cases**
   ```python
   @pytest.mark.parametrize("tier,expected_points", [
       ("Core", 25),
       ("Growth", 15),
       ("Enterprise", 5),
   ])
   def test_tier_points(self, tier, expected_points):
       # Test implementation
   ```

---

## 🎯 Pre-Deployment Checklist

Before deploying to production, ensure:

### P0 Critical
- [ ] All 30 existing tests pass
- [ ] Kill criteria enforced: `test_accuracy_above_kill_criteria` passes
- [ ] Recall constraint met: `test_recall_meets_constraint` passes
- [ ] Schema validation works: `test_scorer_validates_schema` passes
- [ ] Performance acceptable: `test_batch_scoring_performance_10k_clients` passes
- [ ] Error handling clear: `test_missing_required_column_clear_error` passes
- [ ] Integration tests pass: `test_full_experiment_pipeline` passes

### P1 Important
- [ ] Fairness tests pass: `TestModelFairness` all pass
- [ ] Production monitoring: `TestProductionMonitoring` all pass
- [ ] No regressions: `TestPerformanceRegression` all pass
- [ ] Configs valid: `test_all_configs_load` passes

### Documentation
- [ ] Test coverage >80% (`pytest --cov`)
- [ ] All new features have tests
- [ ] Test failures documented
- [ ] Baseline metrics updated if needed

---

## 📚 Key Files

### Schema & Validation
- **`model/schemas.py`** - Pandera schema definitions for input/output validation
- **`model/scorer.py`** - Main scoring logic with schema validation

### Test Files
- **`tests/test_ml_quality.py`** - ML performance, fairness, score quality
- **`tests/test_data_quality.py`** - Schema, ranges, consistency validation
- **`tests/test_production.py`** - Performance, error handling, monitoring
- **`tests/test_regression.py`** - Baselines, determinism, compatibility
- **`tests/test_integration.py`** - End-to-end workflows

### Configuration
- **`pyproject.toml`** - Dependencies and pytest configuration
- **`tests/fixtures/baseline_metrics.json`** - Performance baselines

### Data
- **`experiments/data/train.csv`** - Training data
- **`experiments/data/val.csv`** - Validation data
- **`experiments/data/test.csv`** - Test data

---

## 🔄 Maintenance

### Weekly
- [ ] Review test failures
- [ ] Check experiment logs for patterns
- [ ] Update baseline if legitimate improvements

### Monthly
- [ ] Add tests for production bugs discovered
- [ ] Review and prune obsolete tests
- [ ] Update schema if features added
- [ ] Check test coverage

### Quarterly
- [ ] Audit test coverage gaps
- [ ] Update fairness baselines if tier distribution changes
- [ ] Review and update this guide

---

## 🆘 Getting Help

### Internal Resources
- **This guide:** `TESTING.md`
- **Project context:** `CLAUDE.md`
- **Experiment pipeline:** `docs/diagrams/experiment-pipeline.md`
- **Data analysis:** `EDA-Churn.md`

### Quick Reference Commands
```bash
# List all tests
pytest --collect-only

# Show test coverage
pytest --cov=model --cov=experiments --cov-report=term

# Run specific marker
pytest -m slow  # Run slow tests
pytest -m "not slow"  # Skip slow tests

# Get help
pytest --help
pytest --markers  # Show available markers
```

### Common Issues
1. **Tests failing after data update?** → Check schema validation
2. **Performance regression?** → Profile with `--durations=10`
3. **Fairness issues?** → Review tier distributions
4. **Import errors?** → Check dependencies installed: `pip list | grep -E "pandera|pytest-cov|psutil"`

---

## 🎓 Learning Resources

### Pytest Documentation
- Official docs: https://docs.pytest.org/
- Fixtures: https://docs.pytest.org/en/stable/fixture.html
- Parametrize: https://docs.pytest.org/en/stable/parametrize.html

### Pandera (Schema Validation)
- Official docs: https://pandera.readthedocs.io/
- Examples: https://pandera.readthedocs.io/en/stable/examples.html

### ML Testing Best Practices
- Google ML Testing: https://developers.google.com/machine-learning/testing-debugging
- Test-Driven ML Development: https://martinfowler.com/articles/cd4ml.html

---

**Questions?** Ask Jay Diaz (Director AI/ML) or Xavier Martinez (Tech Lead)

**Last Updated:** 2026-02-01 by Claude Code
**Next Review:** Weekly during Q1 2026 sprint
