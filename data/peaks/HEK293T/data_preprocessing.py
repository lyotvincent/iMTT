

# * sort anchors by chromosome, start and end
def custom_sort(x):
    # sort by int < X < Y
    x = x.replace('chr', '')
    letter_order = {'X': 1, 'Y': 2}  # Add more if needed
    try:
        return int(x), 0
    except ValueError:
        return float('inf'), letter_order.get(x, 0)
# peaks = sorted(peaks, key=lambda x: (custom_sort(x[0]), x[1], x[2]))

def get_intersected_peak(rep1, rep2, rep3):
    """
    Get the intersected peaks from three replicates
    :param rep1: peak file of replicate 1
    :param rep2: peak file of replicate 2
    :param rep3: peak file of replicate 3
    :return: intersected peaks
    """

    def get_chr_start_end_from_bed(rep_path):
        peaks = list()
        with open(rep_path, 'r') as f:
            while line := f.readline().strip().split()[:3]:
                # chr = int(line[0].replace("chr", "").replace("X", "23").replace("Y", "24"))
                if len(line[0]) <= 5:
                    peaks.append([line[0], int(line[1]), int(line[2])])
        return peaks



    # * load & sort peaks
    peaks1 = get_chr_start_end_from_bed(rep1)
    peaks1 = sorted(peaks1, key=lambda x: (custom_sort(x[0]), x[1], x[2]))
    peaks2 = get_chr_start_end_from_bed(rep2)
    peaks2 = sorted(peaks2, key=lambda x: (custom_sort(x[0]), x[1], x[2]))
    peaks3 = get_chr_start_end_from_bed(rep3)
    peaks3 = sorted(peaks3, key=lambda x: (custom_sort(x[0]), x[1], x[2]))
    f = open("./sorted_peaks1.temp", 'w')
    for peak in peaks1:
        f.write(f"{peak[0]}\t{peak[1]}\t{peak[2]}\n")
    f.close()
    f = open("./sorted_peaks2.temp", 'w')
    for peak in peaks2:
        f.write(f"{peak[0]}\t{peak[1]}\t{peak[2]}\n")
    f.close()
    f = open("./sorted_peaks3.temp", 'w')
    for peak in peaks3:
        f.write(f"{peak[0]}\t{peak[1]}\t{peak[2]}\n")
    f.close()
    # print(peaks2)

    # * get intersected peaks
    intersected_peaks = list()
    remaining_peaks1 = list()
    remaining_peaks2 = list()
    remaining_peaks3 = list()
    i, j, k = 0, 0, 0
    while i < len(peaks1) and j < len(peaks2) and k < len(peaks3):
        p1 = peaks1[i]
        p2 = peaks2[j]
        p3 = peaks3[k]
        if p1[0] == p2[0] == p3[0]:
            # * same chr
            # index 0: chr, index 1: start, index 2: end
            if p1[2] < p2[1] or p1[2] < p3[1]:
                remaining_peaks1.append(p1)
                i += 1
            elif p2[2] < p1[1] or p2[2] < p3[1]:
                remaining_peaks2.append(p2)
                j += 1
            elif p3[2] < p1[1] or p3[2] < p2[1]:
                remaining_peaks3.append(p3)
                k += 1
            else:
                intersected_peaks.append([p1[0], max(p1[1], p2[1], p3[1]), min(p1[2], p2[2], p3[2])])
                i += 1
                j += 1
                k += 1
        else:
            # * diff chr
            m = min(custom_sort(p1[0]), custom_sort(p2[0]), custom_sort(p3[0]))
            if custom_sort(p1[0]) == m:
                remaining_peaks1.append(p1)
                i += 1
            elif custom_sort(p2[0]) == m:
                remaining_peaks2.append(p2)
                j += 1
            else:
                remaining_peaks3.append(p3)
                k += 1
    f = open("./intersected_peaks.temp", 'w')
    for peak in intersected_peaks:
        f.write(f"{peak[0]}\t{peak[1]}\t{peak[2]}\n")
    f.close()
    f = open("./remaining_peaks1.temp", 'w')
    for peak in remaining_peaks1:
        f.write(f"{peak[0]}\t{peak[1]}\t{peak[2]}\n")
    f.close()
    f = open("./remaining_peaks2.temp", 'w')
    for peak in remaining_peaks2:
        f.write(f"{peak[0]}\t{peak[1]}\t{peak[2]}\n")
    f.close()
    f = open("./remaining_peaks3.temp", 'w')
    for peak in remaining_peaks3:
        f.write(f"{peak[0]}\t{peak[1]}\t{peak[2]}\n")
    f.close()


def get_interval_regions():
    """
    need sorted peaks files: sorted_peaks1.temp, sorted_peaks2.temp, sorted_peaks3.temp
    """
    with open('./sorted_peaks1.temp', 'r') as f:
        peaks1 = f.readlines()
    with open('./sorted_peaks2.temp', 'r') as f:
        peaks2 = f.readlines()
    with open('./sorted_peaks3.temp', 'r') as f:
        peaks3 = f.readlines()
    peaks1 = [peak.strip().split() for peak in peaks1]
    peaks2 = [peak.strip().split() for peak in peaks2]
    peaks3 = [peak.strip().split() for peak in peaks3]

    def judge_top_peak(p1, p2, p3):
        m = min(custom_sort(p1[0]), custom_sort(p2[0]), custom_sort(p3[0]))
        res = None
        if custom_sort(p1[0]) == m:
            res = p1
            i = 1
        if custom_sort(p2[0]) == m:
            if res==None or p2[1] < res[1]:
                res = p2
                i = 2
        if custom_sort(p3[0]) == m:
            if res==None or p3[1] < res[1]:
                res = p3
                i = 3
        return res, i

    i, j, k = 0, 0, 0
    current_peak = None
    covered_regions = list() # 找到所有三个文件的 peaks 的区域，用于反过来求 interval_regions
    while i < len(peaks1) and j < len(peaks2) and k < len(peaks3):
        p1 = peaks1[i]
        p2 = peaks2[j]
        p3 = peaks3[k]
        if current_peak == None:
            current_peak, ind = judge_top_peak(p1, p2, p3)
            if ind == 1:
                i += 1
            elif ind == 2:
                j += 1
            else:
                k += 1
        else:
            obok = False
            if p1[1] <= current_peak[2]:
                current_peak[2] = max(current_peak[2], p1[2])
                i += 1
                obok = True
            if p2[1] <= current_peak[2]:
                current_peak[2] = max(current_peak[2], p2[2])
                j += 1
                obok = True
            if p3[1] <= current_peak[2]:
                current_peak[2] = max(current_peak[2], p3[2])
                k += 1
                obok = True
            if obok == False:
                covered_regions.append(current_peak)
                current_peak = None

    chr_range = {'chr1':248956422, 'chr2':242193529, 'chr3':198295559, 'chr4':190214555, 'chr5':181538259,
                 'chr6':170805979, 'chr7':159345973, 'chr8':145138636, 'chr9':138394717, 'chr10':133797422,
                 'chr11':135086622, 'chr12':133275309, 'chr13':114364328, 'chr14':107043718, 'chr15':101991189,
                 'chr16':90338345, 'chr17':83257441, 'chr18':80373285, 'chr19':58617616, 'chr20':64444167,
                 'chr21':46709983, 'chr22':50818468, 'chrX':156040895, 'chrY':57227415}
    interval_regions = list()
    chr = 'chr1'
    start = 0
    end = covered_regions[0][1] # 第一个有信号的区域的start
    for cregion in covered_regions:
        if cregion[0] != chr:
            end = chr_range[chr]
            interval_regions.append([chr, start, end])
            chr = cregion[0]
            start = 0
            end = cregion[1]
            interval_regions.append([chr, start, end])
            start = cregion[2]
        else:
            end = cregion[1]
            interval_regions.append([chr, start, end])
            start = cregion[2]
    interval_regions.append([chr, start, chr_range['chrY']])
    
    f = open("./interval_regions.temp", 'w')
    for peak in interval_regions:
        f.write(f"{peak[0]}\t{peak[1]}\t{peak[2]}\n")
    f.close()


            

    


if __name__ == "__main__":
    rep1 = "./GSM7507863_HEK293T_iM_rep1_peaks.stringent.bed"
    rep2 = "./GSM7507864_HEK293T_iM_rep2_peaks.stringent.bed"
    rep3 = "./GSM7507865_HEK293T_iM_rep3_peaks.stringent.bed"
    get_intersected_peak(rep1, rep2, rep3)
    get_interval_regions()