#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/25 0:02
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : scalardata.py
# @Description    : ******
"""

import os
from typing import Callable, List, Union

from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd

from Dataset.data._base import BasicData