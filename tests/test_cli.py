import subprocess
import sys
import hashlib
import pytest
from amateras.cli.bigcellfinder import find_big_cells
from amateras.__main__ import main as amateras_main

# TODO: Test on image_slice_with_hair

PATH = "tests"
IMAGE = PATH + "/img.tif"
OUT_TMP = PATH + "/tmp"
# From 2022-07-15 version 0.1.dev1+ga8edb5e
XY_COORDINATES = [
    (313, 204), (415, 491), (390, 304), (364, 172), (86, 239), (193, 120), (135, 181),
    (63, 121), (435, 452), (282, 18), (94, 332), (313, 160), (89, 446), (15, 403),
    (132, 139), (208, 387), (211, 125), (25, 140), (15, 267), (208, 428), (87, 382),
    (249, 97), (248, 330), (238, 134), (195, 272), (321, 348), (319, 41), (317, 285),
    (131, 102), (370, 472), (192, 431), (152, 442), (234, 75), (375, 291), (163, 168),
    (258, 441), (283, 274), (349, 457), (236, 493), (156, 330), (286, 308), (275, 80),
    (121, 385), (231, 317), (53, 250), (289, 371), (253, 7), (158, 285), (116, 143),
    (256, 35), (3, 229), (243, 308), (342, 230), (143, 348), (177, 470), (232, 16),
    (81, 33), (293, 354), (124, 225), (221, 371), (79, 75), (11, 283), (418, 91),
    (327, 269), (432, 439), (69, 195), (277, 454), (112, 439), (307, 493), (406, 212),
    (312, 19), (344, 406), (48, 463), (121, 460), (372, 449), (146, 309), (408, 259),
    (31, 27), (377, 310), (413, 151), (335, 211), (345, 190), (416, 71), (365, 256),
    (3, 5), (383, 104), (4, 122), (394, 144), (27, 94)
]


@pytest.mark.xfail
def test_main():
    amateras_main()
    return


def get_md5sum(file):
    md5_hash = hashlib.md5()
    with open(file, "rb") as openin:
        for line in openin:
            md5_hash.update(line)
        digest = md5_hash.hexdigest()
    return digest


def test_environment():
    tools = [
        "python --version",
    ]
    for tool in tools:
        print(f"'$ {tool}'")
        subprocess.run(tool.split(" "), stderr=sys.stdout)


def test_big_cell_finder(img=IMAGE, outdir=OUT_TMP):
    xy_coordinates = find_big_cells(input=img, outdir=outdir, n_cells=100)
    print(xy_coordinates)
    assert xy_coordinates == XY_COORDINATES
    return
