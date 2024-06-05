import os
import sys
from sys import argv


def write_file_list_to_file(directory_path, output_file_path):
    """
    Write file list in directory to file.
    """
    try:
        # get file list in directory
        file_list = sorted(os.listdir(directory_path))

        # write file list to file
        with open(output_file_path, 'w', encoding="utf-8") as output_file:
            output_file.write('\n'.join(file_list))

    except OSError as e:
        print(f"File I/O error in write_file_list_to_file(): {e.strerror}")
        sys.exit(1)
    return output_file_path


def read_hitsfile(hitsfile):
    """
    Read hitsfile and return actives list and inactives list.
    """
    f = open(hitsfile, "r", encoding="utf-8").read().split("\n")
    actives = []
    inactives = []
    for line in f:
        if len(line) == 0:
            continue
        try:
            name, hit = line.split(",")
        except ValueError:
            print("Not enough values to unpack:")
            print(line)
            sys.exit(1)
        if int(hit) > 0:
            actives.append(name)
        else:
            inactives.append(name)
    #print(actives, inactives)
    return actives, inactives


def make_labels_txt(train_pngs_file, hitsfile, output_file, header="", test_mode=False):
    """
    Make labels.txt file from train_pngs_file and hitsfile.
    """
    f = open(train_pngs_file, "r", encoding="utf-8").read().split("\n")
    actives, inactives = read_hitsfile(hitsfile)
    o = open(output_file, "w", encoding="utf-8")
    for i, item in enumerate(f):
        if len(item) == 0:
            continue
        cpd = item.split(".")[-2][:-4]

        if cpd in actives:
            label = 1
        elif cpd in inactives:
            label = 0
        elif test_mode:
            label = 0
        else:
            print(f"unknown data: {cpd}")
            continue
        o.write(f"{header}{item} {label}\n")


def main():
    make_labels_txt(*argv[1:])

if __name__ == "__main__":
    main()
