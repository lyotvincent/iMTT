"""
needed features:
[1] C-tract length, 就是下面表格里的 stem_length
[2] iM length, 就是下面表格里的 end
[3] loop length, 
[4] middle loop length, 就是 loop2_length
[5] longest side loop length, 就是 loop1_length 和 loop3_length 中的较大值
[6] shortest side loop length, 就是 loop1_length 和 loop3_length 中的较小值
[7] sum of two side loops, 
[8] longest loop length, 
[9] shortest loop length, 
[10] A density in iMs, 
[11] C density in iMs, 
[12] G density in iMs, 
[13] T density in iMs, 
[14] A density in loops, 
[15] C density in loops, 
[16] G density in loops, 
[17] T density in loops, 
[18] A density in middle loop,
[19] C density in middle loop, 
[20] G density in middle loop, 
[21] T density in middle loop, 
[22] A density in longest side loop, 
[23] C density in longest side loop, 
[24] G density in longest side loop, 
[25] T density in longest side loop, 
[26] A density in shortest side loop, 
[27] C density in shortest side loop, 
[28] G density in shortest side loop, 
[29] T density in shortest side loop, 
[30] A density in two side loops, 
[31] C density in two side loops, 
[32] G density in two side loops, 
[33] T density in two side loops.
"""

"""
columns:
ID      start   end     stem_length     loop1_length    loop2_length    loop3_length    stem1   loop1   stem2   loop2   stem3   loop3   stem4

"""

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.parameters import *

def gen_features(file):
    with open(file, 'r') as f:
        lines = f.readlines()[1:]
    features = []
    for line in lines:
        line = line.strip().split('\t')
        C_tract_len = int(line[3])
        im_len = int(line[2])
        loop_len = int(line[4]) + int(line[5]) + int(line[6])
        middle_loop_length = int(line[5])
        longest_side_loop_length = max(int(line[4]), int(line[6]))
        shortest_side_loop_length = min(int(line[4]), int(line[6]))
        sum_of_two_side_loops = int(line[4]) + int(line[6])
        longest_loop_length = max(int(line[4]), int(line[5]), int(line[6]))
        shortest_loop_length = min(int(line[4]), int(line[5]), int(line[6]))
        iMs = line[7]+line[8]+line[9]+line[10]+line[11]+line[12]+line[13] # is not a feature
        A_density_in_iMs = iMs.count("A") / len(iMs)
        C_density_in_iMs = iMs.count("C") / len(iMs)
        G_density_in_iMs = iMs.count("G") / len(iMs)
        T_density_in_iMs = iMs.count("T") / len(iMs)
        loops = line[8]+line[10]+line[12] # is not a feature
        A_density_in_loops = loops.count("A") / len(loops)
        C_density_in_loops = loops.count("C") / len(loops)
        G_density_in_loops = loops.count("G") / len(loops)
        T_density_in_loops = loops.count("T") / len(loops)
        A_density_in_middle_loop = line[10].count("A") / len(line[10])
        C_density_in_middle_loop = line[10].count("C") / len(line[10])
        G_density_in_middle_loop = line[10].count("G") / len(line[10])
        T_density_in_middle_loop = line[10].count("T") / len(line[10])
        if int(line[4]) >= int(line[6]):
            longest_side_loop = line[8]
            shortest_side_loop = line[12]
        else:
            longest_side_loop = line[12]
            shortest_side_loop = line[8]
        A_density_in_longest_side_loop = longest_side_loop.count("A") / len(longest_side_loop)
        C_density_in_longest_side_loop = longest_side_loop.count("C") / len(longest_side_loop)
        G_density_in_longest_side_loop = longest_side_loop.count("G") / len(longest_side_loop)
        T_density_in_longest_side_loop = longest_side_loop.count("T") / len(longest_side_loop)
        A_density_in_shortest_side_loop = shortest_side_loop.count("A") / len(shortest_side_loop)
        C_density_in_shortest_side_loop = shortest_side_loop.count("C") / len(shortest_side_loop)
        G_density_in_shortest_side_loop = shortest_side_loop.count("G") / len(shortest_side_loop)
        T_density_in_shortest_side_loop = shortest_side_loop.count("T") / len(shortest_side_loop)
        A_density_in_two_side_loops = (line[8]+line[12]).count("A") / len(line[8]+line[12])
        C_density_in_two_side_loops = (line[8]+line[12]).count("C") / len(line[8]+line[12])
        G_density_in_two_side_loops = (line[8]+line[12]).count("G") / len(line[8]+line[12])
        T_density_in_two_side_loops = (line[8]+line[12]).count("T") / len(line[8]+line[12])
        features.append([C_tract_len,
                         im_len,
                         loop_len,
                         middle_loop_length,
                         longest_side_loop_length,
                         shortest_side_loop_length,
                         sum_of_two_side_loops,
                         longest_loop_length,
                         shortest_loop_length,
                         A_density_in_iMs,
                         C_density_in_iMs,
                         G_density_in_iMs,
                         T_density_in_iMs,
                         A_density_in_loops,
                         C_density_in_loops,
                         G_density_in_loops,
                         T_density_in_loops,
                         A_density_in_middle_loop,
                         C_density_in_middle_loop,
                         G_density_in_middle_loop,
                         T_density_in_middle_loop,
                         A_density_in_longest_side_loop,
                         C_density_in_longest_side_loop,
                         G_density_in_longest_side_loop,
                         T_density_in_longest_side_loop,
                         A_density_in_shortest_side_loop,
                         C_density_in_shortest_side_loop,
                         G_density_in_shortest_side_loop,
                         T_density_in_shortest_side_loop,
                         A_density_in_two_side_loops,
                         C_density_in_two_side_loops,
                         G_density_in_two_side_loops,
                         T_density_in_two_side_loops])
    dataset1 = file.split("/")[-1].split(".")[0]
    with open(f"./33_feats/{dataset1}.csv", 'w') as f:
        for feature in features:
            f.write('\t'.join(map(str, feature)) + '\n')
    # return features

def _find_discard_by_imseeker():
    with open("./Putative_iM_Searcher_result_pos_dataset1.txt", 'r') as f:
        lines = f.readlines()[1:]
    gened_list = list()
    for line in lines:
        seq = "".join(line.strip().split()[7:])
        gened_list.append(seq)
    gened_set = set(gened_list)
    print(f"gened_list: {len(gened_list)}, gened_set: {len(gened_set)}")
    with open(SAMPLES_PATH+"/distinct_positive_samples_HEK293T.csv", 'r') as f:
        lines = f.readlines()
    input_list = list()
    for line in lines:
        input_list.append(line.strip())
    input_set = set(input_list)
    dif_set = input_set.difference(gened_set)
    print(f"input_list: {len(input_list)}, input_set: {len(input_set)}")
    print(list(dif_set)[0])
    print(len(dif_set))
# CCCCC AGTCCCGGCGGG CCCC GCGCGCCAGG CCCC TTCCCGACAGG CCCC
# CCC	CCAGT	CCC	GGCGGGC	CCC	GCGCGCCAGGC	CCC
# CCC	CCAGT	CCC	GGCGGGC	CCC	GCGCGCCAGG	CCC



if __name__ == "__main__":
    gen_features("./imsearcher_result/positive_train_samples_dataset1.txt")
    gen_features("./imsearcher_result/positive_train_samples_dataset2.txt")
    gen_features("./imsearcher_result/positive_train_samples_dataset3.txt")
    gen_features("./imsearcher_result/positive_train_samples_dataset4.txt")

    gen_features("./imsearcher_result/negative_train_samples_dataset1.txt")
    gen_features("./imsearcher_result/negative_train_samples_dataset2.txt")
    gen_features("./imsearcher_result/negative_train_samples_dataset3.txt")
    gen_features("./imsearcher_result/negative_train_samples_dataset4.txt")

    gen_features("./imsearcher_result/positive_test_samples_dataset1.txt")
    gen_features("./imsearcher_result/positive_test_samples_dataset2.txt")
    gen_features("./imsearcher_result/positive_test_samples_dataset3.txt")
    gen_features("./imsearcher_result/positive_test_samples_dataset4.txt")

    gen_features("./imsearcher_result/negative_test_samples_dataset1.txt")
    gen_features("./imsearcher_result/negative_test_samples_dataset2.txt")
    gen_features("./imsearcher_result/negative_test_samples_dataset3.txt")
    gen_features("./imsearcher_result/negative_test_samples_dataset4.txt")


    # _find_discard_by_imseeker()


