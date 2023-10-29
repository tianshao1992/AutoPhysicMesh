#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/9/15 14:20
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : python2fmu.py
# @Description    : ******
"""

import os
from abc import abstractmethod
import pandas as pd
from pythonfmu import Fmi2Causality, Fmi2Slave, Fmi2Variability, Boolean, Integer, Real, String

class Python2FMU(Fmi2Slave):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        init func
        | Args
        | __________
        | cols_dict:
        |     配置字典；
        |
        """

        self.window_len = None
        self.input_vars = self.get_inputs()
        self.output_vars = self.get_outputs()
        self._input_dataframe = pd.DataFrame()
        self._output_dataframe = pd.DataFrame()
        self._modes_dict = {'input': Fmi2Causality.input, 'output': Fmi2Causality.output,
                           'param': Fmi2Causality.parameter, 'calculate': Fmi2Causality.calculatedParameter,
                           'local': Fmi2Causality.local}
        self._types_dict = {'bool': Boolean, 'int': Integer, 'str': String,
                           'float': Real, 'real': Real, 'double': Real}
        self._value_dict = {'bool': True, 'int': 0, 'str': '_',
                           'float': 0., 'double': 0., 'real': 0.}
        self._variability_dict = {'int': Fmi2Variability.discrete,
                                  'bool': Fmi2Variability.discrete,
                                  'str': Fmi2Variability.discrete,
                                  'param': Fmi2Variability.tunable,
                                  'float': Fmi2Variability.continuous,
                                  'real': Fmi2Variability.continuous,
                                  'double': Fmi2Variability.continuous}

    def setup_vars(self, window_len=None):
        self.input_vars = self.get_inputs()
        self.output_vars = self.get_outputs()
        self.params_vars = self.get_params()
        self.caculate_vars = self.get_calculate()
        self.local_vars = self.get_locals()
        self.window_len = window_len

    def input_collection(self):
        # self.dataframe =

        df_onestep = self._get_df_step(self.input_vars)
        if self.window_len is None:
            self._input_dataframe = df_onestep
            return self._input_dataframe
        else:
            self._input_dataframe = self._input_dataframe.append(df_onestep)
            self._input_dataframe.reset_index(drop=True, inplace=True)
            if len(self._input_dataframe) > self.window_len:
                self._input_dataframe = self._input_dataframe.drop(index=0)
            self._input_dataframe.reset_index(drop=True, inplace=True)
            return self._input_dataframe

    def output_collection(self, df_outputs):
        self._output_dataframe = df_outputs.copy(deep=True)
        self._data_collection(self.output_vars, self._output_dataframe)
        return True

    def _data_collection(self, vars: list, df: pd.DataFrame):
        self._set_df_step(vars, df)
        return True

    def register_vars(self, df_names, types, modes, re_names=None):

        assert isinstance(df_names, (list, tuple, str)), 'the names must be list or tuple'
        assert isinstance(types, (list, tuple, str)), 'the types must be list or tuple'
        assert isinstance(modes, (list, tuple, str)), 'the types must be list or tuple'

        if re_names is None:
            re_names = df_names.copy()

        if isinstance(df_names, str):
            df_names = [df_names,]
        if isinstance(re_names, str):
            re_names = [re_names,]
        if isinstance(types, str):
            types = [types,] * len(df_names)
        if isinstance(modes, str):
            modes = [modes,] * len(df_names)

        for df_name, re_name, type, mode in zip(df_names, re_names, types, modes):
            re_name = mode + '_' + re_name
            if mode != 'param':
                self.register_variable(self._types_dict[type](re_name, description=df_name,
                                                              causality=self._modes_dict[mode],
                                                              variability=self._variability_dict[type]))
            else:
                self.register_variable(self._types_dict[type](re_name,  description=df_name,
                                                              causality=self._modes_dict[mode],
                                                              variability=Fmi2Variability.tunable))
            if re_name in self.__dict__:
                raise ValueError('the variables name {} has already been defined!'.format(re_name))
            else:

                self.__dict__[re_name] = self._value_dict[type]

    def get_inputs(self):

        inputs = list(
            filter(lambda v: v.causality == Fmi2Causality.input, self.vars.values())
        )

        return inputs

    def get_outputs(self):

        input = list(
            filter(lambda v: v.causality == Fmi2Causality.output, self.vars.values())
        )

        return input

    def get_params(self):

        params = list(
            filter(lambda v: v.causality == Fmi2Causality.parameter, self.vars.values())
        )

        return params

    def get_locals(self):

        locals = list(
            filter(lambda v: v.causality == Fmi2Causality.local, self.vars.values())
        )

        return locals


    def get_calculate(self):

        locals = list(
            filter(lambda v: v.causality == Fmi2Causality.calculatedParameter, self.vars.values())
        )

        return locals

    def _get_df_step(self, vars: list):
        data_dicts = {v.description: getattr(self, v.name) for v in vars}
        df_onestep = pd.DataFrame.from_dict([data_dicts])
        return df_onestep

    def _set_df_step(self, vars: list, df_onestep: pd.DataFrame):

        for v in vars:
            assert v.description in df_onestep.columns.tolist(), 'name: {} not in the dataframe'.format(v.description)
            setattr(self, v.name, df_onestep[v.description].iloc[-1])
        return


    @abstractmethod
    def do_step(self, current_time, step_size):
        pass
        return True
