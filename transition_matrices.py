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
