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


HEADER_KEYWORDS = ["name", "step", "epoch"]
VALUES_KEYWORDS = ["_time", "_learning_rates", "_key_params", "_loss_values", "_metric_values"]

def get_print_keywords(print_dict):
    r"""
        get print keywords
        Args:
            :param print_dict: dict
        :return: list
    """
    head_key_list, value_key_list = [], []
    for key in print_dict.keys():
        for item in HEADER_KEYWORDS:
            if key.endswith(item):
                head_key_list.append(key)
                break
        for item in VALUES_KEYWORDS:
            if key.endswith(item):
                value_key_list.append(key)
                break
    return head_key_list, value_key_list

class Printer(object):
    def __init__(self,
                 config: dict,
                 name: str = "main"):
        r"""
            class to print info
            Args:
                :param config: dict
                :param name: str
        """

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

    def print_info(self, start_time, end_time, print_dict):

        header_keys, values_keys = get_print_keywords(print_dict)

        header_list = [str(key) + ": {}".format(print_dict[key]) for key in header_keys]

        values_list = [[key, "{:.3e}".format(print_dict[key])] for key in values_keys]

        message = tabulate(
            tabular_data=values_list,
            headers=header_list,
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

