## What is a PFSA?
A **Probabilistic Finite State Automaton** is a statistical model that:
- Represents states (ECG leads) and transitions between them
- Captures the probability of moving from one lead to another
- Predicts next fixations based on current position
- Classifies sequences as expert-like or novice-like

**Core Formula**: P(next_lead | current_lead) = transition probability from matrix

---

## Implementation Overview

### Files
- `expert_sample_dataset.json`: 30 expert + 30 novice ECG scanning sequences
- `transition_matrices.py`: Main PFSA implementation with deviation detection
- `statistics_report.py`: Aggregate statistical analysis (optional)

---

## Key Features Implemented

### 1. PFSA Construction with Laplace Smoothing
**Purpose**: Build probabilistic models from training data

**Implementation**:
```python
def build_transition_matrix(sequences, leads, alpha=1.0):
    # Count transitions
    # Apply Laplace smoothing: (count + alpha) / (total + alpha * num_states)
    # Returns 12×12 probability matrix
```

**Why Laplace Smoothing?**
- Prevents zero probabilities for unseen transitions
- More robust generalization from limited training data (30 sequences)
- Better handling of novel but reasonable patterns
- Reduces harsh penalties: unseen transitions get ~2-3% probability instead of 0%

**Result**:
- Expert matrix: 144/144 non-zero transitions (100% dense)
- Novice matrix: 144/144 non-zero transitions (100% dense)

---

### 2. Normalized Log-Likelihood Classification
**Purpose**: Length-independent sequence comparison

**Implementation**:
```python
# Compute per-transition log-likelihood
expert_ll_norm = expert_ll / num_transitions
novice_ll_norm = novice_ll / num_transitions

# Classification
ll_ratio = expert_ll_norm - novice_ll_norm
classification = "Expert-like" if ll_ratio > 0 else "Novice-like"
confidence = abs(ll_ratio)
```

**Why Normalization?**
- Removes sequence length bias
- Fair comparison between sequences of different lengths
- Confidence score represents "per-transition" deviation

**Confidence Interpretation**:
- 0-5: Uncertain, borderline
- 5-10: Moderate confidence
- 10-15: High confidence
- 15+: Very high confidence

---

### 3. Detailed Deviation Detection
**Purpose**: Identify specific problematic transitions

**Features**:
- Detects suboptimal transitions (user choice ≠ expert preference)
- Calculates severity levels: HIGH, MEDIUM, LOW, MINIMAL
- Computes relative quality: how good was your choice vs expert?
- Provides actionable feedback for each deviation

**Severity Thresholds**:
```python
HIGH:    p < 0.01  (less than 1% probability)
MEDIUM:  p < 0.05  (less than 5% probability)
LOW:     p < 0.15  (less than 15% probability)
MINIMAL: p >= 0.15 (15% or higher)
```

**Relative Quality**:
```
Relative Quality = (actual_probability / expert_preference_prob) × 100%
```
- 100%: Perfect match with expert
- 50-99%: Good choice
- 10-49%: Suboptimal
- <10%: Poor choice

---

### 4. Statistical Metrics
**Purpose**: Multi-dimensional sequence analysis

**Per-Sequence Metrics**:
- **Sequence length**: Total number of leads scanned
- **Lead coverage**: % of 12 ECG leads visited
- **Revisit rate**: % of redundant transitions
- **Starting lead**: Typical vs unusual starting position

**Output Format**:
```
Classification: Expert-like (confidence: 2.01)
Length: 13 | Coverage: 100% | Revisit rate: 8%
Suboptimal transitions: 3/12
```

---

## Example Output

```
============================================================
Sequence: advanced_beginner
============================================================
Classification: Expert-like (confidence: 2.01)
Length: 13 | Coverage: 100% | Revisit rate: 8%
Starting lead: Lead_I (typical)
Suboptimal transitions: 3/12

Significant Deviations:
  1. Position 9: V5 → V4
     Actual: p=0.020 [MEDIUM] | Expert prefers: V6 (p=0.776)
     Relative quality: 2.6%
  2. Position 10: V4 → V6
     Actual: p=0.020 [MEDIUM] | Expert prefers: V5 (p=0.776)
     Relative quality: 2.6%
```

---

## Key Improvements Made

### ✅ Laplace Smoothing
- **Before**: 77% sparse expert matrix, harsh unseen transition penalties
- **After**: 100% dense matrices, reasonable penalties for rare transitions

### ✅ Length Normalization
- **Before**: Confidence scores biased by sequence length
- **After**: Fair per-transition comparison across all sequences

### ✅ Detailed Deviation Analysis
- **Before**: Binary classification only
- **After**: Specific feedback on where and why deviations occur

### ✅ Statistical Context
- **Before**: PFSA classification in isolation
- **After**: Combined PFSA + efficiency metrics (coverage, revisits)

---

## Running the Code

```bash
# Main PFSA deviation detection
python3 transition_matrices.py

# Aggregate statistics (optional)
python3 statistics_report.py
```

---

## Use Cases

1. **Real-time Training Feedback**
   - Analyze trainee ECG scanning patterns
   - Provide immediate corrective guidance
   - Track improvement over time

2. **Skill Assessment**
   - Classify scanning behavior as expert-like or novice-like
   - Quantify confidence in classification
   - Identify specific areas for improvement

3. **Pattern Analysis**
   - Understand expert scanning strategies
   - Detect common novice mistakes
   - Design targeted training interventions

---

## Technical Details

**Model Type**: Probabilistic Finite State Automaton (PFSA)
**Training Data**: 30 expert + 30 novice sequences
**States**: 12 ECG leads
**Smoothing**: Laplace (α=1.0)
**Classification**: Normalized log-likelihood ratio
**Metrics**: PFSA confidence + statistical features
