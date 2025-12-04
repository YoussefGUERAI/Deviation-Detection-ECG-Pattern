import numpy as np
import json
from collections import defaultdict

def build_transition_matrix(sequences, leads):
    counts = defaultdict(lambda: defaultdict(int))
    
    for seq in sequences:
        sequence = seq['sequence']
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_lead = sequence[i + 1]
            counts[current][next_lead] += 1
    
    # Convert to probability matrix
    matrix = np.zeros((len(leads), len(leads)))
    lead_to_idx = {lead: i for i, lead in enumerate(leads)}
    
    for i, from_lead in enumerate(leads):
        total = sum(counts[from_lead].values())
        if total > 0:
            for j, to_lead in enumerate(leads):
                matrix[i][j] = counts[from_lead][to_lead] / total
    
    return matrix

leads = ['Lead_I', 'Lead_II', 'Lead_III', 'aVR', 'aVL', 'aVF',
         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Load data from JSON file
with open('expert_sample_dataset.json', 'r') as f:
    data = json.load(f)
    expert_seqs = data[0]  # First array contains expert sequences
    novice_seqs = data[1]  # Second array contains novice sequences

P_expert = build_transition_matrix(expert_seqs, leads)
P_novice = build_transition_matrix(novice_seqs, leads)

# ==================== PFSA Construction ====================

class PFSA:
    """Probabilistic Finite State Automaton for ECG pattern analysis"""
    
    def __init__(self, transition_matrix, states, name="PFSA"):
        """
        Initialize PFSA
        Args:
            transition_matrix: numpy array of transition probabilities
            states: list of state names (ECG leads)
            name: identifier for this PFSA (e.g., "Expert" or "Novice")
        """
        self.transition_matrix = transition_matrix
        self.states = states
        self.name = name
        self.state_to_idx = {state: i for i, state in enumerate(states)} 
        # Map state to index for smooth transition between states and transition matrix 
        self.n_states = len(states)
        
    def get_transition_prob(self, from_state, to_state):
        """Get transition probability from one state to another"""
        i = self.state_to_idx[from_state]
        j = self.state_to_idx[to_state]
        return self.transition_matrix[i][j]
    
    def compute_sequence_log_likelihood(self, sequence):
        """
        Compute log-likelihood of a sequence under this PFSA
        Args:
            sequence: list of states (ECG leads)
        Returns:
            log-likelihood value
        """
        log_likelihood = 0.0
        
        for i in range(len(sequence) - 1):
            prob = self.get_transition_prob(sequence[i], sequence[i + 1])
            if prob > 0:
                log_likelihood += np.log(prob)
            else:
                # Assign very low probability to unseen transitions
                log_likelihood += np.log(1e-10)
        
        return log_likelihood 
    # Compute how probable a sequence is under this PFSA model
    
    def compute_anomaly_score(self, sequence):
        """
        Compute anomaly score (negative log-likelihood)
        Higher score = more anomalous under this PFSA
        """
        return -self.compute_sequence_log_likelihood(sequence)
    
    def display_info(self):
        """Display PFSA information"""
        print(f"\n{'='*50}")
        print(f"PFSA: {self.name}")
        print(f"{'='*50}")
        print(f"Number of states: {self.n_states}")
        print(f"States: {self.states}")
        print(f"\nTransition Matrix Shape: {self.transition_matrix.shape}")
        print(f"Non-zero transitions: {np.count_nonzero(self.transition_matrix)}")
        print(f"Sparsity: {1 - np.count_nonzero(self.transition_matrix) / self.transition_matrix.size:.2%}")

# Create PFSA models
expert_pfsa = PFSA(P_expert, leads, name="Expert")
novice_pfsa = PFSA(P_novice, leads, name="Novice")

# Display information
expert_pfsa.display_info()
novice_pfsa.display_info()

# ==================== Deviation Detection ====================

def analyze_transition_deviations(sequence, expert_pfsa):
    """
    Analyze each transition in the sequence to identify deviations from expert patterns
    
    Args:
        sequence: list of ECG leads
        expert_pfsa: PFSA trained on expert data
    
    Returns:
        list of dictionaries with detailed transition analysis
    """
    deviations = []
    
    for i in range(len(sequence) - 1):
        from_state = sequence[i]
        to_state = sequence[i + 1]
        
        # Get the probability of this transition for experts
        prob = expert_pfsa.get_transition_prob(from_state, to_state)
        
        # Get all possible transitions from this state
        from_idx = expert_pfsa.state_to_idx[from_state]
        all_probs = expert_pfsa.transition_matrix[from_idx]
        
        # Find what experts typically do from this state
        best_next_idx = np.argmax(all_probs)
        best_next_state = expert_pfsa.states[best_next_idx]
        best_prob = all_probs[best_next_idx]
        
        # Calculate deviation severity
        if prob > 0:
            deviation_score = -np.log(prob)  # Lower prob = higher deviation
            is_unseen = False
        else:
            deviation_score = np.inf  # Unseen transition
            is_unseen = True
        
        # Determine severity level
        if is_unseen:
            severity = "CRITICAL"
        elif prob < 0.01:
            severity = "HIGH"
        elif prob < 0.1:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        deviations.append({
            'position': i,
            'transition': f"{from_state} → {to_state}",
            'probability': prob,
            'deviation_score': deviation_score,
            'severity': severity,
            'is_unseen': is_unseen,
            'expert_would_choose': best_next_state,
            'expert_preference_prob': best_prob
        })
    
    return deviations

def analyze_starting_position(sequence, expert_pfsa):
    """
    Analyze if the starting lead is typical for experts
    
    Returns:
        dict with starting position analysis
    """
    start_state = sequence[0]
    
    # Count how often each state appears as first state in training
    # For simplicity, we'll check the most common starting states
    common_starts = ['Lead_I', 'Lead_II']  # Based on expert data
    
    is_common_start = start_state in common_starts
    
    return {
        'starting_lead': start_state,
        'is_typical_start': is_common_start,
        'expected_starts': common_starts,
        'severity': 'LOW' if is_common_start else 'HIGH'
    }

def get_critical_deviations(deviations, top_n=5):
    """Get the most significant deviations"""
    # Sort by deviation score (highest first)
    sorted_devs = sorted(deviations, key=lambda x: x['deviation_score'], reverse=True)
    return sorted_devs[:top_n]

def detect_deviation(sequence, expert_pfsa, novice_pfsa, threshold_ratio=1.0):
    """
    Detect if a sequence deviates from expert behavior
    
    Args:
        sequence: list of ECG leads
        expert_pfsa: PFSA trained on expert data
        novice_pfsa: PFSA trained on novice data
        threshold_ratio: ratio threshold for classification
    
    Returns:
        dict with classification results
    """
    expert_ll = expert_pfsa.compute_sequence_log_likelihood(sequence)
    novice_ll = novice_pfsa.compute_sequence_log_likelihood(sequence)
    
    expert_score = expert_pfsa.compute_anomaly_score(sequence)
    novice_score = novice_pfsa.compute_anomaly_score(sequence)
    
    # Log-likelihood ratio
    ll_ratio = expert_ll - novice_ll
    
    # Classification: positive ratio means more likely expert
    is_expert_like = ll_ratio > threshold_ratio
    
    return {
        'expert_log_likelihood': expert_ll,
        'novice_log_likelihood': novice_ll,
        'log_likelihood_ratio': ll_ratio,
        'expert_anomaly_score': expert_score,
        'novice_anomaly_score': novice_score,
        'classification': 'Expert-like' if is_expert_like else 'Novice-like',
        'confidence': abs(ll_ratio)
    }

def generate_detailed_deviation_report(sequence, expert_pfsa, seq_id="Unknown"):
    """
    Generate a comprehensive deviation report for a sequence
    
    Args:
        sequence: list of ECG leads
        expert_pfsa: PFSA trained on expert data
        seq_id: identifier for the sequence
    
    Returns:
        Prints detailed report and returns analysis data
    """
    print(f"\n{'='*70}")
    print(f"DETAILED DEVIATION REPORT: {seq_id}")
    print(f"{'='*70}")
    
    # Analyze starting position
    start_analysis = analyze_starting_position(sequence, expert_pfsa)
    print(f"\n📍 STARTING POSITION ANALYSIS:")
    print(f"   Started with: {start_analysis['starting_lead']}")
    print(f"   Typical expert starts: {', '.join(start_analysis['expected_starts'])}")
    if not start_analysis['is_typical_start']:
        print(f"   ⚠️  WARNING: Unusual starting position (Severity: {start_analysis['severity']})")
    else:
        print(f"   ✅ Good: Started with a typical expert lead")
    
    # Analyze all transitions
    deviations = analyze_transition_deviations(sequence, expert_pfsa)
    
    # Get critical deviations
    critical = get_critical_deviations(deviations, top_n=5)
    
    print(f"\n🔍 CRITICAL DEVIATIONS (Top 5 Most Significant):")
    print(f"{'─'*70}")
    
    for idx, dev in enumerate(critical, 1):
        print(f"\n   {idx}. Position {dev['position']}: {dev['transition']}")
        print(f"      Severity: {dev['severity']}")
        if dev['is_unseen']:
            print(f"      ❌ NEVER seen in expert training data!")
        else:
            print(f"      Probability: {dev['probability']:.4f}")
            print(f"      Deviation Score: {dev['deviation_score']:.2f}")
        print(f"      💡 Expert would typically choose: {dev['expert_would_choose']} (prob: {dev['expert_preference_prob']:.4f})")
    
    # Summary statistics
    total_transitions = len(deviations)
    critical_count = sum(1 for d in deviations if d['severity'] in ['CRITICAL', 'HIGH'])
    unseen_count = sum(1 for d in deviations if d['is_unseen'])
    avg_prob = np.mean([d['probability'] for d in deviations if not d['is_unseen']])
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total transitions: {total_transitions}")
    print(f"   Critical/High severity: {critical_count} ({critical_count/total_transitions*100:.1f}%)")
    print(f"   Unseen in expert data: {unseen_count}")
    print(f"   Average transition probability: {avg_prob:.4f}")
    
    print(f"\n{'='*70}\n")
    
    return {
        'start_analysis': start_analysis,
        'deviations': deviations,
        'critical_deviations': critical,
        'stats': {
            'total_transitions': total_transitions,
            'critical_count': critical_count,
            'unseen_count': unseen_count,
            'avg_probability': avg_prob
        }
    }

def analyze_dataset(sequences, label, expert_pfsa, novice_pfsa):
    """Analyze a dataset of sequences"""
    print(f"\n{'='*60}")
    print(f"Analyzing {label} Sequences")
    print(f"{'='*60}")
    
    results = []
    for seq_data in sequences[:5]:  # Analyze first 5 as examples
        seq = seq_data['sequence']
        result = detect_deviation(seq, expert_pfsa, novice_pfsa)
        results.append(result)
        
        print(f"\nSequence ID: {seq_data['id']}")
        print(f"  True Label: {seq_data['label']}")
        print(f"  Classification: {result['classification']}")
        print(f"  LL Ratio: {result['log_likelihood_ratio']:.2f}")
        print(f"  Expert LL: {result['expert_log_likelihood']:.2f}")
        print(f"  Novice LL: {result['novice_log_likelihood']:.2f}")
    
    return results

# ==================== Analysis ====================

print("\n" + "="*60)
print("PFSA-Based ECG Pattern Deviation Detection")
print("="*60)

# Analyze expert sequences
expert_results = analyze_dataset(expert_seqs, "Expert", expert_pfsa, novice_pfsa)

# Analyze novice sequences  
novice_results = analyze_dataset(novice_seqs, "Novice", expert_pfsa, novice_pfsa)

print("\n" + "="*60)
print("PFSA models built successfully!")
print("="*60)

# ==================== Detailed Deviation Analysis ====================

print("\n" + "="*70)
print("DETAILED DEVIATION ANALYSIS")
print("="*70)

# Analyze a few example sequences in detail
print("\n🔬 EXPERT SEQUENCE EXAMPLE:")
generate_detailed_deviation_report(expert_seqs[0]['sequence'], expert_pfsa, expert_seqs[0]['id'])

print("\n🔬 NOVICE SEQUENCE EXAMPLES:")
# Analyze first 2 novice sequences to show different deviation patterns
for i in range(2):
    generate_detailed_deviation_report(novice_seqs[i]['sequence'], expert_pfsa, novice_seqs[i]['id'])

print("\n" + "="*70)
print("Analysis Complete!")
print("="*70)
