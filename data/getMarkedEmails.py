#!/usr/bin/env python
import sys

def main():
    fin_name = "MarkedEmail.txt"
    foutS_name = "MarkedSignatures.txt"
    foutB_name = "EmailBody.txt"
    try:
        fin = open(fin_name, 'r')
        foutS = open(foutS_name, 'w')
        foutB = open(foutB_name, 'w')
        for line in fin:
            if "#sig#" in line:
                term = line.split('#')
                foutS.write(term[-1])
            else:
                if len(line.split())>0:
                    foutB.write(line + '\n')
        fin.close()
        foutB.close()
        foutS.close()
    except IOError:
        print "Files fail to open"
        sys.exit(-1)

if __name__ == "__main__":
    main()