from scipy.stats import entropy
import json
import numpy as np
from transition_matrices import P_expert, P_novice

def compute_metrics(sequences, trans_matrix):
    # Transition entropy (average uncertainty)
    row_entropies = []
    for row in trans_matrix:
        if row.sum() > 0:
            row_entropies.append(entropy(row, base=2))
    avg_entropy = np.mean(row_entropies)
    
    # Revisit rate
    revisits = 0
    total_trans = 0
    for seq in sequences:
        sequence = seq['sequence']
        for i in range(len(sequence) - 1):
            if sequence[i] == sequence[i + 1]:
                revisits += 1
            total_trans += 1
    revisit_rate = revisits / total_trans if total_trans > 0 else 0
    
    # Average sequence length
    avg_length = np.mean([len(s['sequence']) for s in sequences])
    
    return {
        'entropy': avg_entropy,
        'revisit_rate': revisit_rate,
        'avg_length': avg_length
    }
# Load data from JSON file
with open('expert_sample_dataset.json', 'r') as f:
    data = json.load(f)
    expert_seqs = data[0]  # First array contains expert sequences
    novice_seqs = data[1]  # Second array contains novice sequences
    
expert_metrics = compute_metrics(expert_seqs, P_expert)
novice_metrics = compute_metrics(novice_seqs, P_novice)

print("Expert entropy:", expert_metrics['entropy'])
print("Novice entropy:", novice_metrics['entropy'])
print("Expert revisit rate:", expert_metrics['revisit_rate'])
print("Novice revisit rate:", novice_metrics['revisit_rate'])
print("Expert average sequence length:", expert_metrics['avg_length'])
print("Novice average sequence length:", novice_metrics['avg_length'])


