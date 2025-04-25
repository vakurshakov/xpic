#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from lib.common import makedirs
from configuration import const

os.chdir(const.output_path)

if len(sys.argv) == 1:
    plots = (
        "fields",
        "info_ions",
        "info_electrons",
        "currents",
    )
else:
    plots = sys.argv[1:]

makedirs(f"{const.output_path}/video")

for plot in plots:
    if os.path.exists(plot):
        os.system(f"ffmpeg -y -i ./{plot}/%0{len(str(const.Nt))}d.png -r 15 ./video/{plot}.mp4")