#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/18 11:19
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : logger.py
# @Description    : ******
"""
import os
import logging
from tabulate import tabulate


def get_log_keys(log_dict):
    key_list = []
    for key in log_dict.keys():
        if key.endswith("_loss_values"):
            key_list.append(key)
        elif key.endswith("_metric_values"):
            key_list.append(key)
        elif key.endswith("_learning_rates"):
            key_list.append(key)
    return key_list


class Logger:
    def __init__(self, name: str = "main"):
        self.logger = logging.getLogger(name)
        self.logger.handlers.clear()
        formatter = logging.Formatter(
            "[%(asctime)s - %(name)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )

        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        streamhandler.setLevel(logging.INFO)
        self.logger.addHandler(streamhandler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(message)

    def log_print(self, step, start_time, end_time, log_dict):
        log_keys = get_log_keys(log_dict)

        log_list = [[key, "{:.3e}".format(log_dict[key])] for key in log_keys]

        message = tabulate(
            log_list,
            headers=[
                "Epoch : {:3d}".format(step),
                "Time  : {:.3f}".format(end_time - start_time),
            ],
            tablefmt="simple",
            numalign="right",
            disable_numparse=True,
        )

        header_length = len(message.split("\n")[0]) + 2
        dashed_line = "-" * header_length
        message = dashed_line + "\n" + message

        for line in message.split("\n"):
            self.logger.info(line)

    # def log_visual_mesh(self, mesh):

