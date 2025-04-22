[![Windows MacOS Ubuntu](https://github.com/FrickTobias/amateras/actions/workflows/tests.yml/badge.svg)](https://github.com/FrickTobias/amateras/actions/workflows/tests.yml)

**Note:** This is not under active development. Therefore the fail above has not been dealt with. However, the code most likely works without issues. 

# AMATERAS
<img src=process-description.gif width="500" title="process" alt="detection process" align="center" vspace = "50">
Image analysis for AMATERAS microscope

## Setup

    git clone https://github.com/FrickTobias/amateras.git
    cd amateras
    pip install -e .

## Update

    cd amateras
    git pull

## Usage

    amateras bigcellfinder -h

## Examples

Minimal

    amateras bigcellfinder tests/img.tif --n-cells 100 > ~/Desktop/positions.txt

Write qc output

    amateras bigcellfinder tests/img.tif --n-cells 100 --qc-outdir ~/Desktop/bigcells-qc-output > ~/Desktop/positions.txt

Include details of all steps in analysis

    amateras bigcellfinder tests/img.tif --n-cells 100 --qc-outdir ~/Desktop/bigcells-qc-output --details > ~/Desktop/positions.txt

## Advanced examples 
Try modifying program settings if cells are missing or being falsely identified

Use automatic thresholding

    amateras bigcellfinder tests/img.tif --auto-thresh --n-cells 100 > ~/Desktop/positions.txt

Set thresholding manually (corresponds to highest px value for black area of cell lowest px value for white area of cell)

    amateras bigcellfinder tests/img.tif --black-thresh 90 --white-thresh 150 --n-cells 100 > ~/Desktop/positions.txt

Filter out high density regions

    amateras bigcellfinder tests/img.tif --final-filter --n-cells 100 > ~/Desktop/positions.txt

