#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from configuration import output_path, Nt

os.chdir(output_path)

if len(sys.argv) == 1:
    plots = (
        "fields",
        "info_ions",
        "info_electrons",
        "currents",
    )
else:
    plots = sys.argv[1:]

for plot in plots:
    os.system(f"ffmpeg -y -i ./{plot}/%0{len(str(Nt))}d.png -r 15 ./video/{plot}.mp4")