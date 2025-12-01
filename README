## What is a PDFA?
A **Probabilistic Deterministic Finite Automaton** is a statistical model that:
- Represents states (ECG leads) and transitions between them
- Captures the probability of moving from one lead to another
- Predicts next fixations based on current position
- Classifies sequences as expert-like or novice-like

**Core Formula**: P(next_lead | current_lead) = transition probability from matrix

---

## Step 1: Data Preparation & Analysis (COMPLETED)

### 1.1 Load and Parse Dataset
**Purpose**: Data ingestion and validation

**Tasks**:
- Load JSON file with 60 sequences
- Validate structure (sequences, durations, labels)
- Separate expert (30) and novice (30) sequences

**Output**: Two clean arrays ready for analysis

---

### 1.2 Build Transition Matrices ⭐ **CORE STEP**
**Purpose**: Extract sequential patterns - THIS IS THE PDFA MODEL

**Tasks**:
- Count all Lead_X → Lead_Y transitions across all sequences
- Build 12×12 count matrices for each class
- Normalize to probabilities (row sums = 1.0)

**Example**:
```
From Lead_II in expert sequences:
- Lead_I: 67% (20 out of 30 times)
- Lead_III: 25% (7 out of 30 times)
- aVR: 8% (3 out of 30 times)
```

**Output**: 
- **Expert transition matrix**: P_expert[12×12]
- **Novice transition matrix**: P_novice[12×12]

**Why both matrices?**
- To **compare** expert systematic patterns vs novice randomness
- To **classify** new sequences: which model fits better?
- To **quantify** behavioral differences statistically

---

### 1.3 Compute Statistical Metrics
**Purpose**: Quantify behavioral differences with numbers

#### Metric 1: **Transition Entropy** (Predictability)
- **Measures**: Randomness in transition choices
- **Formula**: H = -Σ p(x) × log₂(p(x))
- **Expert**: ~1.5-2.0 bits (LOW = predictable, systematic)
- **Novice**: ~2.5-3.5 bits (HIGH = unpredictable, random)
- **Value**: Statistical proof experts are more systematic

#### Metric 2: **Sequence Diversity** (Strategy Consistency)
- **Measures**: Number of unique scanning patterns
- **Expert**: ~45 unique trigrams from 30 sequences (LOW = shared strategy)
- **Novice**: ~150 unique trigrams from 30 sequences (HIGH = everyone improvising)
- **Value**: Shows experts learned a common "expert way"

#### Metric 3: **Revisit Rate** (Efficiency)
- **Measures**: % of transitions returning to already-seen leads
- **Expert**: ~10-15% (LOW = efficient, purposeful re-checks)
- **Novice**: ~25-40% (HIGH = getting lost, confusion)
- **Value**: Distinguishes strategic verification from disorientation

#### Metric 4: **Lead Coverage** (Attention Allocation)
- **Measures**: Which leads get viewed and for how long
- **Expert**: 100% coverage of critical leads (Lead_II, V3, V4)
- **Novice**: ~60% coverage, misses important leads
- **Value**: Identifies training gaps (what novices skip)

**Output**: Statistical table for research paper:
```
┌──────────────────┬─────────┬─────────┬─────────┐
│ Metric           │ Expert  │ Novice  │ p-value │
├──────────────────┼─────────┼─────────┼─────────┤
│ Entropy (bits)   │ 1.85    │ 3.12    │ <0.001  │
│ Unique Patterns  │ 48      │ 156     │ <0.001  │
│ Revisit Rate (%) │ 12.3    │ 34.7    │ <0.001  │
│ Critical Cov (%) │ 98.2    │ 62.4    │ <0.001  │
└──────────────────┴─────────┴─────────┴─────────┘
```
