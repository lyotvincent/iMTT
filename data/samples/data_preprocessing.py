

import re, os
from Bio import SeqIO

def match_imotif_from_seq(sequence):
    # Define the regular expression for identifying i-motif sequences with lookahead for overlapping matches
    # 第一个模式用于定位可能的i-motif序列的开始位置，
    pattern1 = r"(?=(C{3,}[ATCG]{1,12}){3}C{3,})"
    pattern1 = re.compile(pattern1)

    # Perform the matching
    matches = pattern1.finditer(sequence)
    matches = [m.start() for m in pattern1.finditer(sequence)]

    # Output the matches
    # print("Matching i-motifs:", matches)

    # Define the regular expression for identifying i-motif sequences
    # 而第二个模式则用于从这些位置提取完整的i-motif序列。
    pattern2 = r"(C{3,}[ATCG]{1,12}){3}C{3,}"
    pattern2 = re.compile(pattern2)

    results = list()
    for start in matches:
        # Perform the matching
        match = pattern2.match(sequence, start)

        # Output the matches
        # print("Matching i-motifs:", match)
        results.append(match.group(0))
    return results

def get_hg38():
    HG38_PATH = os.path.join(ROOT_PATH, "data", 'hg38', 'hg38.fa')
    chr_dict = dict()
    for seq_record in SeqIO.parse(HG38_PATH, "fasta"):
        if len(seq_record.id) <= 5:
            print(f"loading {seq_record.id}")
            chr_dict[seq_record.id] = seq_record.seq.upper()
    return chr_dict

def get_regions(peaks_path):
    candidates = list()
    with open(peaks_path, 'r') as f:
        while line := f.readline():
            chr, start, end = line.strip().split()
            try:
                sequence = str(hg38_chr_dict[chr][int(start)-1:int(end)+1])
                sequence = sequence.upper()
            except:
                print(chr)
                print(hg38_chr_dict.keys())
            # check all bases is ATCG
            if re.match(r'^[ATCG]+$', sequence):
                candidates.extend(match_imotif_from_seq(sequence))
    return candidates

def samples_gen(cell_line):
    INTERSECTED_PEAKS_PATH = os.path.join(PEAK_PATH, cell_line, "intersected_peaks.temp")
    positive_imotifs = get_regions(INTERSECTED_PEAKS_PATH)
    print(f"intersected peaks number: {len(positive_imotifs)}, distinct imotifs number: {len(set(positive_imotifs))}")
    positive_imotifs = set(positive_imotifs)

    REMAINING_PEAKS_PATH1 = os.path.join(PEAK_PATH, cell_line, 'remaining_peaks1.temp')
    unlabeled_samples1 = get_regions(REMAINING_PEAKS_PATH1)
    print(f"unlabeled_samples1 number: {len(unlabeled_samples1)}, distinct imotifs number: {len(set(unlabeled_samples1))}")
    unlabeled_samples1 = set(unlabeled_samples1)

    REMAINING_PEAKS_PATH2 = os.path.join(PEAK_PATH, cell_line, 'remaining_peaks2.temp')
    unlabeled_samples2 = get_regions(REMAINING_PEAKS_PATH2)
    print(f"unlabeled_samples2 number: {len(unlabeled_samples2)}, distinct imotifs number: {len(set(unlabeled_samples2))}")
    unlabeled_samples2 = set(unlabeled_samples2)

    REMAINING_PEAKS_PATH3 = os.path.join(PEAK_PATH, cell_line, 'remaining_peaks3.temp')
    unlabeled_samples3 = get_regions(REMAINING_PEAKS_PATH3)
    print(f"unlabeled_samples3 number: {len(unlabeled_samples3)}, distinct imotifs number: {len(set(unlabeled_samples3))}")
    unlabeled_samples3 = set(unlabeled_samples3)
    unlabeled_samples = unlabeled_samples1.union(unlabeled_samples2).union(unlabeled_samples3)
    print(f"unlabeled_samples num: {len(unlabeled_samples)}")

    INTERVAL_REGIONS_PATH = os.path.join(PEAK_PATH, cell_line, 'interval_regions.temp')
    negative_samples = get_regions(INTERVAL_REGIONS_PATH)
    print(f"negative_samples number: {len(negative_samples)}, distinct imotifs number: {len(set(negative_samples))}")
    negative_samples = set(negative_samples)

    # with open(os.path.join(SAMPLES_PATH, f"positive_samples_{cell_line}.csv"), 'w') as f:
    #     for i in positive_imotifs:
    #         f.write(i+"\n")
    # with open(os.path.join(SAMPLES_PATH, f"unlabeled_samples1_{cell_line}.csv"), 'w') as f:
    #     for i in unlabeled_samples1:
    #         f.write(i+"\n")
    # with open(os.path.join(SAMPLES_PATH, f"unlabeled_samples2_{cell_line}.csv"), 'w') as f:
    #     for i in unlabeled_samples2:
    #         f.write(i+"\n")
    # with open(os.path.join(SAMPLES_PATH, f"unlabeled_samples3_{cell_line}.csv"), 'w') as f:
    #     for i in unlabeled_samples3:
    #         f.write(i+"\n")
    # with open(os.path.join(SAMPLES_PATH, f"negative_samples_{cell_line}.csv"), 'w') as f:
    #     for i in negative_samples:
    #         f.write(i+"\n")

    # 获取在
    intersection1 = positive_imotifs.intersection(negative_samples)
    intersection2 = positive_imotifs.intersection(unlabeled_samples)
    intersection3 = negative_samples.intersection(unlabeled_samples)
    intersection = intersection1.union(intersection2).union(intersection3)
    print(f"same seq num: {len(intersection)}")

    distinct_positive_samples = positive_imotifs.difference(intersection)
    distinct_unlabeled_samples = unlabeled_samples.difference(intersection)
    distinct_negative_samples = negative_samples.difference(intersection)
    print(f"distinct pos samples num: {len(distinct_positive_samples)}")
    print(f"distinct unlabeled samples num: {len(distinct_unlabeled_samples)}")
    print(f"distinct neg samples num: {len(distinct_negative_samples)}")

    with open(os.path.join(SAMPLES_PATH, f"distinct_positive_samples_{cell_line}.csv"), 'w') as f:
        for i in distinct_positive_samples:
            f.write(i+"\n")
    with open(os.path.join(SAMPLES_PATH, f"distinct_unlabeled_samples_{cell_line}.csv"), 'w') as f:
        for i in distinct_unlabeled_samples:
            f.write(i+"\n")
    with open(os.path.join(SAMPLES_PATH, f"distinct_negative_samples_{cell_line}.csv"), 'w') as f:
        for i in distinct_negative_samples:
            f.write(i+"\n")


if __name__ == "__main__":
    # Example sequence
    sequence = "CCCTTTTATTGTTTTCCCTTTTATTGTTTTCCCTTTTATTGTTTTCCCTTTTATTGTTTTCCCTTTTATTGTTTTCCCTTTTATTGTTTTCCCTTTTATTGTTTTCCCTTTTATTGTTTTCCC"
    # match_imotif_from_seq(sequence)
    # exit()

    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # print(ROOT_PATH)
    PEAK_PATH = os.path.join(ROOT_PATH, "data", "peaks")
    SAMPLES_PATH = os.path.join(ROOT_PATH, "data", "samples")

    hg38_chr_dict = get_hg38()
    samples_gen('HEK293T')
    samples_gen('WDLPS')

