import numpy as np
import json
import time
from collections import defaultdict

def calculate_sequence_stats(sequence):
    """Calculate basic statistics for a single sequence"""
    unique_leads = len(set(sequence))
    sequence_length = len(sequence)
    lead_coverage = unique_leads / 12
    revisit_rate = (sequence_length - unique_leads) / sequence_length if sequence_length > 0 else 0
    
    return {
        'sequence_length': sequence_length,
        'unique_leads': unique_leads,
        'lead_coverage': lead_coverage,
        'revisit_rate': revisit_rate
    }

def build_transition_matrix(sequences, leads, alpha=1.0):
    """
    Build transition probability matrix with Laplace smoothing
    
    Args:
        sequences: list of sequence dictionaries
        leads: list of state names
        alpha: Laplace smoothing parameter (default=1.0)
    
    Returns:
        Smoothed transition probability matrix
    """
    counts = defaultdict(lambda: defaultdict(int))
    
    for seq in sequences:
        sequence = seq['sequence']
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_lead = sequence[i + 1]
            counts[current][next_lead] += 1
    
    # Convert to probability matrix with Laplace smoothing
    matrix = np.zeros((len(leads), len(leads)))
    lead_to_idx = {lead: i for i, lead in enumerate(leads)}
    num_states = len(leads)
    
    for i, from_lead in enumerate(leads):
        total = sum(counts[from_lead].values())
        for j, to_lead in enumerate(leads):
            # Laplace smoothing: add alpha to count, add (alpha * num_states) to total
            matrix[i][j] = (counts[from_lead][to_lead] + alpha) / (total + alpha * num_states)
    
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
            # With Laplace smoothing, all probabilities are > 0
            log_likelihood += np.log(prob)
        
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
        

# Create PFSA models
expert_pfsa = PFSA(P_expert, leads, name="Expert")
novice_pfsa = PFSA(P_novice, leads, name="Novice")

# Display information
expert_pfsa.display_info()
novice_pfsa.display_info()


# Up to this point , we built the transition matrices and can compute log-likelihoods for sequences.

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
        
        # Calculate deviation score (lower prob = higher deviation)
        deviation_score = -np.log(prob)
        
        # Calculate how much worse this choice is compared to expert preference
        if best_next_state != to_state:
            prob_ratio = prob / best_prob
            deviation_magnitude = best_prob - prob
        else:
            prob_ratio = 1.0
            deviation_magnitude = 0.0
        
        # Determine severity level based on probability
        if prob < 0.01:
            severity = "HIGH"
        elif prob < 0.05:
            severity = "MEDIUM"
        elif prob < 0.15:
            severity = "LOW"
        else:
            severity = "MINIMAL"
        
        deviations.append({
            'position': i,
            'transition': f"{from_state} → {to_state}",
            'probability': prob,
            'deviation_score': deviation_score,
            'severity': severity,
            'expert_would_choose': best_next_state,
            'expert_preference_prob': best_prob,
            'is_suboptimal': (best_next_state != to_state),
            'prob_ratio': prob_ratio,
            'deviation_magnitude': deviation_magnitude
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

def get_significant_deviations(deviations, top_n=5):
    """Get the most significant deviations (only suboptimal choices)"""
    # Filter only suboptimal transitions
    suboptimal = [d for d in deviations if d['is_suboptimal']]
    
    # Sort by deviation magnitude (how much worse than expert choice)
    sorted_devs = sorted(suboptimal, key=lambda x: x['deviation_magnitude'], reverse=True)
    return sorted_devs[:top_n]

def detect_deviation(sequence, expert_pfsa, novice_pfsa):
    """
    Detect if a sequence deviates from expert behavior
    
    Args:
        sequence: list of ECG leads
        expert_pfsa: PFSA trained on expert data
        novice_pfsa: PFSA trained on novice data
    
    Returns
        dict with classification results
    """
    # Calculate raw log-likelihoods
    expert_ll = expert_pfsa.compute_sequence_log_likelihood(sequence)
    novice_ll = novice_pfsa.compute_sequence_log_likelihood(sequence)
    
    # Normalize by sequence length to remove length bias
    num_transitions = len(sequence) - 1
    expert_ll_norm = expert_ll / num_transitions
    novice_ll_norm = novice_ll / num_transitions
    
    expert_score = expert_pfsa.compute_anomaly_score(sequence)
    novice_score = novice_pfsa.compute_anomaly_score(sequence)
    
    # Use normalized log-likelihood ratio for classification
    ll_ratio_norm = expert_ll_norm - novice_ll_norm
    
    # Classification: positive ratio means more likely expert
    is_expert_like = ll_ratio_norm > 0
    
    return {
        'expert_log_likelihood': expert_ll,
        'novice_log_likelihood': novice_ll,
        'expert_ll_normalized': expert_ll_norm,
        'novice_ll_normalized': novice_ll_norm,
        'log_likelihood_ratio': ll_ratio_norm,
        'expert_anomaly_score': expert_score,
        'novice_anomaly_score': novice_score,
        'classification': 'Expert-like' if is_expert_like else 'Novice-like',
        'confidence': abs(ll_ratio_norm)
    }

def analyze_sequence(sequence, expert_pfsa, novice_pfsa, seq_id="Unknown"):
    """
    Unified sequence analysis combining classification and deviation detection
    
    Args:
        sequence: list of ECG leads
        expert_pfsa: PFSA trained on expert data
        novice_pfsa: PFSA trained on novice data
        seq_id: identifier for the sequence
    
    Returns:
        Combined analysis results
    """
    # Classification
    result = detect_deviation(sequence, expert_pfsa, novice_pfsa)
    
    # Deviation analysis
    deviations = analyze_transition_deviations(sequence, expert_pfsa)
    significant = get_significant_deviations(deviations, top_n=5)
    start_analysis = analyze_starting_position(sequence, expert_pfsa)
    
    # Statistical features
    stats = calculate_sequence_stats(sequence)
    
    # Calculate deviation stats
    total_transitions = len(deviations)
    suboptimal_count = sum(1 for d in deviations if d['is_suboptimal'])
    
    # Print comprehensive report
    print(f"\n{'='*60}")
    print(f"Sequence: {seq_id}")
    print(f"{'='*60}")
    print(f"Classification: {result['classification']} (confidence: {result['confidence']:.2f})")
    print(f"Length: {stats['sequence_length']} | Coverage: {stats['lead_coverage']*100:.0f}% | Revisit rate: {stats['revisit_rate']*100:.0f}%")
    print(f"Starting lead: {start_analysis['starting_lead']} ({'typical' if start_analysis['is_typical_start'] else 'unusual'})")
    print(f"Suboptimal transitions: {suboptimal_count}/{total_transitions}")
    
    if len(significant) > 0:
        print(f"\nSignificant Deviations:")
        for idx, dev in enumerate(significant, 1):
            print(f"  {idx}. Position {dev['position']}: {dev['transition']}")
            print(f"     Actual: p={dev['probability']:.3f} [{dev['severity']}] | Expert prefers: {dev['expert_would_choose']} (p={dev['expert_preference_prob']:.3f})")
            print(f"     Your choice is {dev['prob_ratio']*100:.1f}% as likely as expert's preferred choice.")
    else:
        print(f"\nNo significant deviations detected!")
    
    print()
    
    return {
        'classification': result['classification'],
        'confidence': result['confidence'],
        'suboptimal_count': suboptimal_count,
        'total_transitions': total_transitions,
        'significant_deviations': significant,
        'stats': stats
    }

# ==================== Test with New Challenging Sequences ====================

print("\n" + "="*60)
print("TESTING SOME CHALLENGING SEQUENCES")
print("="*60)

# Advanced Beginner: Mostly follows expert pattern but makes a few mistakes
advanced_beginner = ["Lead_I", "Lead_II", "Lead_III", "aVR", "aVL", "aVF", 
                     "V1", "V2", "V3", "V5", "V4", "V6", "Lead_II"]
# Mistake: V3 → V5 instead of V3 → V4

# Intermediate: Systematic but with some inefficient revisits
intermediate = ["Lead_II", "Lead_I", "Lead_III", "aVR", "aVL", "aVF",
                "V1", "V2", "V3", "V4", "V5", "V6", 
                "Lead_II", "aVL", "V3", "V4"]
# Has extra revisits but follows general pattern

# Semi-Expert: Perfect start but loses structure in precordial leads
semi_expert = ["Lead_I", "Lead_II", "Lead_III", "aVR", "aVL", "aVF",
               "V1", "V3", "V2", "V4", "V6", "V5"]
# Out of order in V-leads: V3 before V2, V6 before V5

# Confused: Mixes limb and precordial leads chaotically
confused = ["Lead_I", "V1", "Lead_II", "V3", "aVR", "V5", 
            "Lead_III", "V2", "aVL", "V4", "aVF", "V6"]
# Constantly jumping between limb and precordial

# Almost Expert: Perfect except one subtle error
almost_expert = ["Lead_I", "Lead_II", "Lead_III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6", "aVF", "V5"]
# Only issue: revisits at end (aVF, V5) instead of typical Lead_II

print("\nADVANCED BEGINNER:")
analyze_sequence(advanced_beginner, expert_pfsa, novice_pfsa, "advanced_beginner")

print("\nINTERMEDIATE:")
analyze_sequence(intermediate, expert_pfsa, novice_pfsa, "intermediate")

print("\nSEMI-EXPERT:")
analyze_sequence(semi_expert, expert_pfsa, novice_pfsa, "semi_expert")

print("\nCONFUSED:")
analyze_sequence(confused, expert_pfsa, novice_pfsa, "confused")

print("\nALMOST EXPERT:")
analyze_sequence(almost_expert, expert_pfsa, novice_pfsa, "almost_expert")

# ==================== COMPLEXITY VALIDATION - TIMING EXPERIMENTS ====================

print("\n" + "="*70)
print("COMPLEXITY VALIDATION - TIMING EXPERIMENTS")
print("="*70)
print("\nMeasuring execution times for complexity analysis in paper...")

# Time training (average over 100 iterations)
print("\n1. Training Complexity (Theorem 3)")
print("-" * 50)
start = time.time()
for _ in range(100):
    P_expert_test = build_transition_matrix(expert_seqs, leads)
    P_novice_test = build_transition_matrix(novice_seqs, leads)
training_time = (time.time() - start) / 100
print(f"Training time (m=60, n̄=15): {training_time*1000:.2f}ms")
print(f"Expected: O(m·n̄ + |Q|²) = O(60·15 + 144) = O(1044)")

# Time classification (average over 1000 iterations)
print("\n2. Inference Complexity (Theorem 2)")
print("-" * 50)
test_seq = advanced_beginner
start = time.time()
for _ in range(1000):
    result = detect_deviation(test_seq, expert_pfsa, novice_pfsa)
classification_time = (time.time() - start) / 1000
print(f"Classification time (n=13): {classification_time*1000:.3f}ms")
print(f"Throughput: {1/classification_time:.0f} sequences/second")
print(f"Expected: O(n) = O(13)")

# Time deviation detection (average over 1000 iterations)
print("\n3. Deviation Detection Complexity (Theorem 4)")
print("-" * 50)
start = time.time()
for _ in range(1000):
    deviations = analyze_transition_deviations(test_seq, expert_pfsa)
deviation_time = (time.time() - start) / 1000
print(f"Deviation detection time (n=13): {deviation_time*1000:.2f}ms")
print(f"Expected: O(n·|Q|) = O(13·12) = O(156)")

# Total analysis time
print("\n4. Total Analysis Performance")
print("-" * 50)
total_time = classification_time + deviation_time
print(f"Total per sequence: {total_time*1000:.2f}ms")
print(f"Throughput: {1/total_time:.0f} sequences/second")
print(f"Real-time feasibility: {'YES' if total_time < 0.016 else 'NO'} (< 16ms for 60 FPS)")

# Scalability experiments
print("\n5. Scalability Analysis")
print("-" * 50)

# Test with longer sequence
long_seq = ["Lead_I", "Lead_II", "Lead_III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6"] * 2  # 24 leads
start = time.time()
for _ in range(1000):
    result = detect_deviation(long_seq, expert_pfsa, novice_pfsa)
long_time = (time.time() - start) / 1000
print(f"Long sequence (n=24): {long_time*1000:.3f}ms")
print(f"Ratio (24/13): {long_time/classification_time:.2f}x (linear scaling)")

# Process 1000 sequences
start = time.time()
for _ in range(1000):
    result = detect_deviation(test_seq, expert_pfsa, novice_pfsa)
    deviations = analyze_transition_deviations(test_seq, expert_pfsa)
batch_time = time.time() - start
print(f"\nProcess 1000 sequences: {batch_time:.3f}s = {batch_time*1000:.0f}ms")
print(f"Average per sequence: {batch_time:.3f}ms")

print("\n" + "="*70)
print("TIMING SUMMARY FOR PAPER (Table in Section 3.5)")
print("="*70)
print(f"{'Operation':<35} {'Time':<15} {'Throughput':<20}")
print("-" * 70)
print(f"{'Train PDFA (m=60, n̄=15)':<35} {training_time*1000:.1f}ms{'':<10} {'—':<20}")
print(f"{'Classify sequence (n=13)':<35} {classification_time*1000:.2f}ms{'':<9} {f'{1/classification_time:.0f}/sec':<20}")
print(f"{'Deviation detection (n=13)':<35} {deviation_time*1000:.2f}ms{'':<9} {f'{1/deviation_time:.0f}/sec':<20}")
print(f"{'Full analysis':<35} {total_time*1000:.2f}ms{'':<9} {f'{1/total_time:.0f}/sec':<20}")
print("-" * 70)
print(f"{'Process 1000 sequences':<35} {batch_time*1000:.0f}ms{'':<9} {'—':<20}")
print("="*70)

print("\n✓ All complexity measurements complete!")
print("✓ Copy the timing summary table into Section 3.5 of your paper")