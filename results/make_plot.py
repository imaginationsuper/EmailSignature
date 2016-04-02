#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 2:
        print "Invalid argument!"
        sys.exit(-1)
    model_name = sys.argv[1]
    filename = model_name + "-ValidationStats.csv"
    try:
        fp = open(filename, 'r')
        index = 0
        for line in fp:
            index += 1
            if index == 3:
                dataset_ns = [ds.strip() for ds in line.split(',') if ds!='\n']
            elif index == 5:
                fmedian_ns = [float(fm.strip()) for fm in line.split(',') if fm!='\n']
            elif index == 11:
                iqrf_ns = [float(fv.strip()) for fv in line.split(',') if fv!='\n']
            elif index == 18:
                dataset_ws = [ds.strip() for ds in line.split(',') if ds != '\n']
            elif index == 20:
                fmedian_ws = [float(fm.strip()) for fm in line.split(',') if fm != '\n']
            elif index == 26:
                iqrf_ws = [float(fv.strip()) for fv in line.split(',') if fv != '\n']
        fp.close()
        fmedian_diff = list()
        for index in range(len(fmedian_ns)):
            fmedian_diff.append(abs(fmedian_ws[index] - fmedian_ns[index]))
        sorted_indices = sorted(range(len(fmedian_diff)), key=lambda i: fmedian_diff[i])
        ds_ns, fm_ns, iqf_ns, ds_ws, fm_ws, iqf_ws = list(), list(), list(), list(), list(), list()
        for index in sorted_indices:
            ds_ns.append(dataset_ns[index])
            fm_ns.append(fmedian_ns[index])
            iqf_ns.append(iqrf_ns[index])
            ds_ws.append(dataset_ws[index])
            fm_ws.append(fmedian_ws[index])
            iqf_ws.append(iqrf_ws[index])
        saveplot_name = model_name + "-ValidationStatsPlot.png"
        plt.figure()
        numOfBins = len(ds_ns)
        dataBinIndex = [i for i in range(numOfBins)]
        plt.plot(dataBinIndex, fm_ns, color="blue", linewidth=2, label="non-SMOTE_median")
        plt.plot(dataBinIndex, fm_ws, color="green", linewidth=2, label="SMOTE_median")
        plt.plot(dataBinIndex, iqf_ns, color="blue", linestyle='--', linewidth=2,
                 label="non-SMOTE_iqr")
        plt.plot(dataBinIndex, iqf_ws, color="green", linestyle='--', linewidth=2, label="SMOTE_iqr")
        plt.xticks(dataBinIndex, ds_ns, rotation='vertical')
        plt.ylim([0, 1.0])
        plt.legend(loc=0, borderaxespad=0.5, frameon=False)
        plt.ylabel("F-scores")
        plt.xlabel("Data Bin")
        plt.savefig(saveplot_name)
    except IOError:
        print "File fails to open!"
        sys.exit(-1)

if __name__ == "__main__":
    main()
