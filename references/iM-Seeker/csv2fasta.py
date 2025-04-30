import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.parameters import *

def csv2fasta(csv_file, fasta_file):
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    with open(fasta_file, 'w') as f:
        for i, line in enumerate(lines):
            line = line.strip()
            f.write('>seq' + str(i) + '\n')
            f.write(line + '\n')

if __name__ == '__main__':
    # csv2fasta(os.path.join(SAMPLES_PATH, 'distinct_positive_samples_HEK293T.csv'), './distinct_positive_samples_HEK293T.fasta')
    # csv2fasta(os.path.join(SAMPLES_PATH, 'distinct_negative_samples_HEK293T.csv'), './distinct_negative_samples_HEK293T.fasta')

    # csv2fasta('./origin_data/positive_train_samples.csv', './fasta_data/positive_train_samples.fasta')
    # csv2fasta('./origin_data/negative_train_samples.csv', './fasta_data/negative_train_samples.fasta')
    # csv2fasta('./origin_data/positive_test_samples.csv', './fasta_data/positive_test_samples.fasta')
    # csv2fasta('./origin_data/negative_test_samples.csv', './fasta_data/negative_test_samples.fasta')

    for i in range(5):
        csv2fasta(f'./origin_data/positive_train_samples_fold{i}.csv', f'./fasta_data/positive_train_samples_fold{i}.fasta')
        csv2fasta(f'./origin_data/negative_train_samples_fold{i}.csv', f'./fasta_data/negative_train_samples_fold{i}.fasta')
        csv2fasta(f'./origin_data/positive_test_samples_fold{i}.csv', f'./fasta_data/positive_test_samples_fold{i}.fasta')
        csv2fasta(f'./origin_data/negative_test_samples_fold{i}.csv', f'./fasta_data/negative_test_samples_fold{i}.fasta')