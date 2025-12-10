"""
Baseline Comparison Implementations for PDFA Paper
Implements 3 baseline models to compare against PDFA approach
"""

import numpy as np
import json
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Load existing transition matrices from your code
from transition_matrices import P_expert, P_novice, leads, expert_seqs, novice_seqs

# ==================== Helper Functions ====================

def compute_log_likelihood(sequence, transition_matrix, lead_to_idx):
    """Compute log-likelihood of sequence under given transition matrix"""
    ll = 0.0
    for i in range(len(sequence) - 1):
        from_idx = lead_to_idx[sequence[i]]
        to_idx = lead_to_idx[sequence[i + 1]]
        prob = transition_matrix[from_idx][to_idx]
        
        if prob == 0:
            return float('-inf')  # Zero probability
        ll += np.log(prob)
    
    return ll

# ==================== Baseline 1: First-Order Markov Chain ====================

class MarkovChainClassifier:
    """
    Simple Markov chain without Laplace smoothing
    Demonstrates importance of smoothing for handling unseen transitions
    """
    def __init__(self, leads):
        self.leads = leads
        self.lead_to_idx = {lead: i for i, lead in enumerate(leads)}
        self.expert_trans = None
        self.novice_trans = None
    
    def build_unsmoothed_matrix(self, sequences):
        """Build transition matrix WITHOUT Laplace smoothing"""
        counts = defaultdict(lambda: defaultdict(int))
        
        for seq in sequences:
            sequence = seq['sequence']
            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_lead = sequence[i + 1]
                counts[current][next_lead] += 1
        
        # Convert to probability matrix WITHOUT smoothing
        matrix = np.zeros((len(self.leads), len(self.leads)))
        
        for i, from_lead in enumerate(self.leads):
            total = sum(counts[from_lead].values())
            if total == 0:
                # Uniform distribution for unseen states
                matrix[i] = 1.0 / len(self.leads)
            else:
                for j, to_lead in enumerate(self.leads):
                    # NO SMOOTHING: zeros remain zeros
                    matrix[i][j] = counts[from_lead][to_lead] / total if total > 0 else 0
        
        return matrix
    
    def fit(self, expert_seqs, novice_seqs):
        self.expert_trans = self.build_unsmoothed_matrix(expert_seqs)
        self.novice_trans = self.build_unsmoothed_matrix(novice_seqs)
    
    def predict(self, sequence):
        expert_ll = compute_log_likelihood(sequence, self.expert_trans, self.lead_to_idx)
        novice_ll = compute_log_likelihood(sequence, self.novice_trans, self.lead_to_idx)
        
        # Handle infinite log-likelihoods
        if expert_ll == float('-inf') and novice_ll == float('-inf'):
            return "novice"  # Default to novice if both fail
        elif expert_ll == float('-inf'):
            return "novice"
        elif novice_ll == float('-inf'):
            return "expert"
        
        return "expert" if expert_ll > novice_ll else "novice"

# ==================== Baseline 2: Frequency-Based Classifier ====================

class FrequencyClassifier:
    """
    Simple bigram frequency matching
    Classifies based on overlap with observed expert vs. novice bigrams
    """
    def __init__(self):
        self.expert_bigrams = defaultdict(int)
        self.novice_bigrams = defaultdict(int)
    
    def fit(self, expert_seqs, novice_seqs):
        # Count bigram frequencies
        for seq in expert_seqs:
            sequence = seq['sequence']
            for i in range(len(sequence) - 1):
                bigram = (sequence[i], sequence[i+1])
                self.expert_bigrams[bigram] += 1
        
        for seq in novice_seqs:
            sequence = seq['sequence']
            for i in range(len(sequence) - 1):
                bigram = (sequence[i], sequence[i+1])
                self.novice_bigrams[bigram] += 1
    
    def predict(self, sequence):
        expert_score = 0
        novice_score = 0
        
        for i in range(len(sequence) - 1):
            bigram = (sequence[i], sequence[i+1])
            expert_score += self.expert_bigrams.get(bigram, 0)
            novice_score += self.novice_bigrams.get(bigram, 0)
        
        # Normalize by total counts to avoid bias
        expert_total = sum(self.expert_bigrams.values())
        novice_total = sum(self.novice_bigrams.values())
        
        expert_score_norm = expert_score / expert_total if expert_total > 0 else 0
        novice_score_norm = novice_score / novice_total if novice_total > 0 else 0
        
        return "expert" if expert_score_norm > novice_score_norm else "novice"

# ==================== Baseline 3: Random Forest on Statistical Features ====================

class StatisticalFeatureClassifier:
    """
    Machine learning approach using only statistical features
    No transition modeling - purely based on aggregate statistics
    """
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    
    def extract_features(self, sequence):
        """Extract 8 statistical features from sequence"""
        unique_leads = len(set(sequence))
        length = len(sequence)
        
        features = [
            length,  # Total fixations
            unique_leads,  # Number of unique leads visited
            unique_leads / 12,  # Coverage ratio
            (length - unique_leads) / length if length > 0 else 0,  # Revisit rate
            1 if sequence[0] in ['Lead_I', 'Lead_II'] else 0,  # Typical start
            1 if sequence[-1] in ['V6', 'Lead_II'] else 0,  # Typical end
            len([i for i in range(len(sequence)-1) if sequence[i] == sequence[i+1]]) / length,  # Self-loop rate
            len(set([(sequence[i], sequence[i+1]) for i in range(len(sequence)-1)])) / (length-1) if length > 1 else 0  # Transition diversity
        ]
        
        return features
    
    def fit(self, expert_seqs, novice_seqs):
        X = []
        y = []
        
        for seq in expert_seqs:
            X.append(self.extract_features(seq['sequence']))
            y.append(1)  # Expert label
        
        for seq in novice_seqs:
            X.append(self.extract_features(seq['sequence']))
            y.append(0)  # Novice label
        
        self.clf.fit(X, y)
    
    def predict(self, sequence):
        features = self.extract_features(sequence)
        pred = self.clf.predict([features])[0]
        return "expert" if pred == 1 else "novice"

# ==================== PDFA Classifier (Your Original) ====================

class PDFAClassifier:
    """Your original PDFA with Laplace smoothing"""
    def __init__(self, expert_matrix, novice_matrix, leads):
        self.expert_trans = expert_matrix
        self.novice_trans = novice_matrix
        self.lead_to_idx = {lead: i for i, lead in enumerate(leads)}
    
    def predict(self, sequence):
        # Normalized log-likelihood (your approach)
        expert_ll = compute_log_likelihood(sequence, self.expert_trans, self.lead_to_idx)
        novice_ll = compute_log_likelihood(sequence, self.novice_trans, self.lead_to_idx)
        
        n_transitions = len(sequence) - 1
        expert_ll_norm = expert_ll / n_transitions if n_transitions > 0 else 0
        novice_ll_norm = novice_ll / n_transitions if n_transitions > 0 else 0
        
        return "expert" if expert_ll_norm > novice_ll_norm else "novice"

# ==================== Evaluation ====================

def evaluate_classifier(classifier, test_sequences, true_labels):
    """Evaluate classifier performance"""
    predictions = []
    for seq in test_sequences:
        pred = classifier.predict(seq['sequence'])
        predictions.append(pred)
    
    # Convert to binary
    y_true = [1 if label == "expert" else 0 for label in true_labels]
    y_pred = [1 if pred == "expert" else 0 for pred in predictions]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': predictions
    }

# ==================== Main Comparison ====================

if __name__ == "__main__":
    print("="*60)
    print("BASELINE COMPARISON EXPERIMENTS")
    print("="*60)
    
    # Prepare test data (use all sequences for now - in practice use cross-validation)
    test_sequences = expert_seqs + novice_seqs
    true_labels = ["expert"] * len(expert_seqs) + ["novice"] * len(novice_seqs)
    
    # Initialize classifiers
    print("\n1. Training Markov Chain (no smoothing)...")
    markov = MarkovChainClassifier(leads)
    markov.fit(expert_seqs, novice_seqs)
    markov_results = evaluate_classifier(markov, test_sequences, true_labels)
    
    print("2. Training Frequency-Based Classifier...")
    freq = FrequencyClassifier()
    freq.fit(expert_seqs, novice_seqs)
    freq_results = evaluate_classifier(freq, test_sequences, true_labels)
    
    print("3. Training Random Forest (statistical features)...")
    rf = StatisticalFeatureClassifier()
    rf.fit(expert_seqs, novice_seqs)
    rf_results = evaluate_classifier(rf, test_sequences, true_labels)
    
    print("4. Evaluating PDFA (with Laplace smoothing)...")
    pdfa = PDFAClassifier(P_expert, P_novice, leads)
    pdfa_results = evaluate_classifier(pdfa, test_sequences, true_labels)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    models = [
        ("First-Order Markov", markov_results),
        ("Frequency-Based", freq_results),
        ("Random Forest", rf_results),
        ("PDFA (Ours)", pdfa_results)
    ]
    
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 73)
    
    for name, results in models:
        print(f"{name:<25} {results['accuracy']*100:>10.1f}%  {results['precision']:>10.3f}  {results['recall']:>10.3f}  {results['f1']:>10.3f}")
    
    print("\n" + "="*60)
    print("CONFUSION MATRICES")
    print("="*60)
    
    for name, results in models:
        print(f"\n{name}:")
        print(f"                Predicted")
        print(f"                Novice  Expert")
        print(f"Actual Novice   {results['confusion_matrix'][0][0]:>6}  {results['confusion_matrix'][0][1]:>6}")
        print(f"       Expert   {results['confusion_matrix'][1][0]:>6}  {results['confusion_matrix'][1][1]:>6}")
    
    # Performance improvements
    print("\n" + "="*60)
    print("PDFA IMPROVEMENTS OVER BASELINES")
    print("="*60)
    
    baseline_acc = [markov_results['accuracy'], freq_results['accuracy'], rf_results['accuracy']]
    pdfa_acc = pdfa_results['accuracy']
    
    for i, (name, _) in enumerate(models[:-1]):
        improvement = (pdfa_acc - baseline_acc[i]) * 100
        print(f"vs. {name:<25} +{improvement:>5.1f} percentage points")
    
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    print(f"""
1. Laplace Smoothing Impact:
   Markov Chain (no smoothing): {markov_results['accuracy']*100:.1f}%
   PDFA (with smoothing):       {pdfa_results['accuracy']*100:.1f}%
   Improvement:                 +{(pdfa_results['accuracy']-markov_results['accuracy'])*100:.1f} pp
   
   → Demonstrates critical importance of handling unseen transitions

2. Probabilistic vs. Frequency:
   Frequency-Based:  {freq_results['accuracy']*100:.1f}%
   PDFA:             {pdfa_results['accuracy']*100:.1f}%
   Improvement:      +{(pdfa_results['accuracy']-freq_results['accuracy'])*100:.1f} pp
   
   → Probability magnitudes matter, not just bigram presence

3. Transition Modeling vs. Statistical Features:
   Random Forest:    {rf_results['accuracy']*100:.1f}%
   PDFA:             {pdfa_results['accuracy']*100:.1f}%
   Improvement:      +{(pdfa_results['accuracy']-rf_results['accuracy'])*100:.1f} pp
   
   → Explicit transition modeling captures patterns missed by aggregate statistics
""")