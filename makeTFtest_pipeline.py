#!/usr/bin/env python3

"""
Make tfrecords file from maegz file. Do not require hitsfile.
argv: visinet_dir job_dir input_maegz jobid
"""
import sys
import os
from sys import argv
from pathlib import Path
from makeTFtrain_pipeline import make_png, make_tfrecord, remove_png_files


def main(visinet_dir, job_dir, input_maegz, jobid,
         remove_png=False):
    """
    Main function.
    """
    tfrecord_path = f"{job_dir}/tfrecord/test_{jobid}.tfrecord"
    os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
    labels_path = f"{job_dir}/labels/test_{jobid}"
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    input_maegz_filename = Path(input_maegz).stem

    # exit if final result already exists
    if os.path.exists(tfrecord_path):
        print(f"Exist TFrecord {tfrecord_path} . exit.")
        sys.exit()

    # Make png
    print("making png files...", flush=True)
    make_png(visinet_dir, input_maegz, job_dir, mode="test")
    print("Done.", flush=True)
    # png file:
    # {job_dir}/test/{input_maegz_filename}.lig_{0??}.png
    # ??: 00-80

    # Make test labels file
    print("Making test labels file from png files...", flush=True)
    sorted_files = sorted(Path(f"{job_dir}/test").glob(
        f"{input_maegz_filename}.*_???.png"))
    with open(labels_path, "w", encoding="utf-8") as label_file:
        for f in sorted_files:
            label_file.write(f"{f} 0\n")
    print("Done.", flush=True)

    # Make test tf file
    print("Making tfrecords file...", flush=True)
    make_tfrecord(visinet_dir, jobid,
                  tfrecord_path, labels_path)

    print("Done.", flush=True)

    if remove_png:
        print("Removing png files...", flush=True)
        remove_png_files(job_dir, "test",
                         f"{input_maegz_filename}*_???.png")

    print("All processes have finished.", flush=True)


if __name__ == "__main__":
    usage = """
    Usage: python makeTFtest_pipeline.py \
visinet_dir job_dir \
input_maegz jobid
    """
    if len(argv) < 5:
        print(usage)
        sys.exit(1)

    _visinet_dir = argv[1]
    _job_dir = argv[2]
    _input_maegz = argv[3]
    _jobid = argv[4]
    print(sys.argv)
    main(_visinet_dir, _job_dir, _input_maegz, _jobid,
         remove_png=False)
