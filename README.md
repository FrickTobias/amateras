[![Windows MacOS Ubuntu](https://github.com/FrickTobias/amateras/actions/workflows/tests.yml/badge.svg)](https://github.com/FrickTobias/amateras/actions/workflows/tests.yml)

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

