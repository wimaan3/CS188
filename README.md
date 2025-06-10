# CS188 Final Project: ORB-SLAM & RRT

## Overview

This repository contains our final project for CS188, showcasing an autonomous navigation pipeline integrating ORB-SLAM for mapping and RRT for path planning.

## Repository Structure

```
├── orbtest.py           # ORB-SLAM test on a pre-recorded video
├── simulate_slam.py     # Run SLAM on a live video or images to generate trajectories and landmarks
├── visual_slam1.py      # Run visual 3D slam to generate the trajectories and landmarks
├── visual_slam2.py      # Load the made trajectories and landmarks to test RRT on the 3D map
├── convert_to_pgm.py    # Convert 2D or 3D png files to PGM and generate corresponding yaml file
├── rrt_map2d.py         # RRT planner on 2D Slam map
├── rrt_map3d.py         # RRT planner using 3D SLAM outputs
├── check.py             # Checkerboard calibration generator for intrinsic and extrinsic matricies 
├── images/              # Contains checkerboard calibration image, slam png map generations
├── npy/                 # Saved numpy arrays (maps, trajectories)
├── pgm/                 # PGM map files for planners
├── yaml/                # Map metadata in YAML format
├── project_report.pdf   # Final project report
├── index.html           # Project website
├── requirements.txt     # Python dependencies
└── README.md            # This document
```

## Prerequisites

* Python 3.8 or higher
* OpenCV
* NumPy
* Matplotlib
* scikit-image

Dependencies are listed in **`requirements.txt`**.

## Installation

```bash
git clone https://github.com/wimaan3/CS188.git
cd CS188
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. SLAM Mapping

Generate intrinsic and extrinsic matricies using a checkerboard image and your camera of the device you run this from
Change the matrix values in the corresponding files
```bash
python check.py 
```

Generate SLAM maps from a video or image sequence, change file paths accordingly

```bash
python simulate_slam.py 
```

Generate resulting maps in pgm form change paths in file accordingly:

```bash
python convert_to_pgm.py 
```

Run RRT on PGM map choose start and goal and see simulation run

```bash
python rrt_map2d.py
```


## Configuration

Parameter files in **`yaml/`** define map metadata and planner settings. Edit these YAML files to adjust:

* SLAM feature counts, scale levels, etc.
* RRT step sizes, maximum iterations, goal biases, etc.

## Videos

Check these videos for instructions:
https://www.icloud.com/photos/#/icloudlinks/0dfTQ50FrlYUPwPG9NzAPZ-Vw

https://www.icloud.com/photos/#/icloudlinks/0abfEnWkPKtfCW9sU73CIzwxQ