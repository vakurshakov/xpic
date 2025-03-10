#!/usr/bin/env python3

import json

config = None

with open("../config.json", "r") as file:
    config = json.load(file)
    file.close()
