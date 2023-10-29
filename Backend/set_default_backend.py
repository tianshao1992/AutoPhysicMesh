#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/3 12:50
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : set_default_backend.py
# @Description    : This file is used to set default backend.
"""


import argparse
import json
import os

def set_default_backend(backend_name):
    """
    Set default backend.
    :param backend_name: backend name
    """
    default_dir = os.path.join(os.path.expanduser("~"), ".mpo")
    if not os.path.exists(default_dir):
        os.makedirs(default_dir)
    config_path = os.path.join(default_dir, "config.json")
    with open(config_path, "w") as config_file:
        json.dump({"backend": backend_name.lower()}, config_file)
    print(
        'Setting the default backend to "{}". You can change it in the '
        "~/.mpo/config.json file or export the MPO_BACKEND environment variable. "
        "Valid options are: pytorch, taichi, jax, paddle (all lowercase)".format(
            backend_name
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        nargs=1,
        default="pytorch",
        type=str,
        choices=["pytorch", "taichi", "jax", "paddle"],
        help="Set default backend",
    )
    args = parser.parse_args()
    set_default_backend(args.backend)
