# 数据处理过程

(1) transfer bigwig to bedgraph (/data/bigwig/bigwig2bedgraph.sh)
(2) (/data/bedgraph/seacr.sh)
(3) 获取三个样本重叠区域，和不重叠区域。 (/peaks/HEK293T/data_preprocessing.py)
> GSM7507863_HEK293T_iM_rep1_peaks.stringent.bed -> sorted_peaks1.temp -> remaining_peaks1.temp + intersected_peaks.temp  
GSM7507864_HEK293T_iM_rep2_peaks.stringent.bed -> sorted_peaks2.temp -> remaining_peaks2.temp + intersected_peaks.temp  
GSM7507865_HEK293T_iM_rep3_peaks.stringent.bed -> sorted_peaks3.temp -> remaining_peaks3.temp + intersected_peaks.temp  

> intersected peaks number: 15357, distinct imotifs number: 14160  
unlabeled_samples1 number: 8257, distinct imotifs number: 7888  
unlabeled_samples2 number: 20705, distinct imotifs number: 19882  
unlabeled_samples3 number: 11420, distinct imotifs number: 10890  
negative_samples number: 31889, distinct imotifs number: 29896  
intersected peaks number: 12, distinct imotifs number: 12  
unlabeled_samples1 number: 651, distinct imotifs number: 409  
unlabeled_samples2 number: 167, distinct imotifs number: 164  
unlabeled_samples3 number: 211, distinct imotifs number: 172  
negative_samples number: 3358, distinct imotifs number: 3243  

> The focus was concentrated on HEK293T cell data which was presented with more high-confident iM regions than WDLPS cells (. Zanin, I. , Ruggiero, E. , Nicoletto, G. , Lago, S. , Maurizio, I. , Gallina, I. and Richter,S.N. (2023) Genome-wide mapping of i-motifs reveals their association with transcription regulation in live human cells. Nucleic Acids Res., 51 , 8309–8321.)