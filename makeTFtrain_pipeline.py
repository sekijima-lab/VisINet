#!/usr/bin/env python3

"""
Make tfrecords file from maegz file.
argv: visinet_dir job_dir input_maegz hitsfile jobid
"""
import sys
import os
import subprocess
import glob
from sys import argv
from pathlib import Path
from make_labels_txt import write_file_list_to_file, make_labels_txt


def make_png(visinet_dir, input_maegz, job_dir, mode="train"):
    """
    Make png files from maegz file.
    Loop until the process is successful.
    """
    os.makedirs(f"{job_dir}/{mode}/", exist_ok=True)
    while True:
        result = subprocess.run([
            f"{visinet_dir}/getimage.py", input_maegz,
            f"{job_dir}/{mode}/", "9", "-d"], check=False)
        if result.returncode == 0:
            break


def make_tfrecord(visinet_dir, jobid,
                  tfrecord_path, labels_path):
    """
    Make tfrecords file from png files and labels file.
    """
    os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
    subprocess.run(["./makeTF_wrapper.sh",
                    visinet_dir, labels_path, tfrecord_path],
                   check=True)
    if not os.path.exists(tfrecord_path):
        print(f"Failed to make {tfrecord_path}.", flush=True)
        sys.exit(1)
    else:
        print(f"Successfully made {tfrecord_path}.", flush=True)


def remove_png_files(job_dir, subdir, png_pattern):
    """
    Remove png files in {job_dir}/{subdir}/{png_pattern}.
    """
    for f in glob.glob(f"{job_dir}/{subdir}/{png_pattern}"):
        os.remove(f)


def main(visinet_dir, job_dir,
         input_maegz, hitsfile, jobid,
         remove_png=False):
    """
    Main function.
    """
    tfrecord_path = f"{job_dir}/tfrecord/train_{jobid}.tfrecord"
    os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
    labels_path = f"{job_dir}/labels/train_{jobid}"
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    input_maegz_filename = Path(input_maegz).stem

    # exit if final result already exists
    if os.path.exists(tfrecord_path):
        print(f"Exist TFrecord {tfrecord_path} . exit.")
        sys.exit()

    # Make png
    print("making png files...", flush=True)
    make_png(visinet_dir, input_maegz, job_dir, mode="train")
    print("Done.", flush=True)
    # png file:
    # {job_dir}/train/{input_maegz_filename}.lig_{000-080}.png like that

    # Make train labels file
    print("Making train labels file from png files and hit definition...",
          flush=True)
    png_filelist = f"{job_dir}/train_png.txt"
    write_file_list_to_file(f"{job_dir}/train/", png_filelist)
    make_labels_txt(png_filelist, hitsfile,
                    labels_path, header=f"{job_dir}/train/")
    os.remove(png_filelist)
    print("Done.", flush=True)

    # Make train tf file
    print("Making tfrecords file...", flush=True)
    make_tfrecord(visinet_dir, jobid,
                  tfrecord_path, labels_path)

    print("Done.", flush=True)

    if remove_png:
        print("Removing png files...", flush=True)
        remove_png_files(job_dir, "train",
                         f"{input_maegz_filename}*_???.png")

    print("All processes have finished.", flush=True)


if __name__ == "__main__":
    usage = """
    Usage:
        python makeTFtrain_pipeline.py \
            visinet_dir job_dir \
            input_maegz hitsfile jobid
    """
    if len(argv) < 6:
        print(usage)
        sys.exit(1)

    _visinet_dir = argv[1]
    _job_dir = argv[2]
    _input_maegz = argv[3]
    _hitsfile = argv[4]
    _jobid = argv[5]
    print(sys.argv)
    main(_visinet_dir, _job_dir, _input_maegz, _hitsfile, _jobid,
         remove_png=False)
