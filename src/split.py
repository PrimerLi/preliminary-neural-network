def printLines(header, lines, outputFileName):
    ofile = open(outputFileName, "w")
    ofile.write(header + "\n")
    for i in range(len(lines)):
        ofile.write(lines[i] + "\n")
    ofile.close()

def split(inputFileName, trainRatio):
    import os
    assert(os.path.exists(inputFileName))
    lines = []
    header = ""
    ifile = open(inputFileName, "r")
    for (index, string) in enumerate(ifile):
        if (index == 0):
            header = string.strip("\n")
        else:
            lines.append(string.strip("\n"))
    ifile.close()
    
    trainNumber = int(len(lines)*trainRatio)
    printLines(header, lines[0:trainNumber], "train.csv")
    printLines(header, lines[trainNumber:], "test.csv")

def main():
    import sys
    if (len(sys.argv) != 3):
        print "inputFileName = sys.argv[1], trainRatio = sys.argv[2]. "
        return -1

    inputFileName = sys.argv[1]
    trainRatio = float(sys.argv[2])
    assert(trainRatio > 0 and trainRatio < 1)
    split(inputFileName, trainRatio)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
