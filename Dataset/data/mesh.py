#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/09/15 1:30
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : meshdata.py
# @Description    : ******
"""

import os
import gmsh
import meshio
import numpy as np
from Dataset.data._base_data import BasicData

class MeshData(BasicData, meshio.Mesh):
    def __init__(self, file):
        mesh_model = self.load_file(file)
        super(MeshData, self).__init__(
                points=mesh_model.points,
                cells=mesh_model.cells,
                point_data=mesh_model.point_data,
                cell_data=mesh_model.cell_data,
                field_data=mesh_model.field_data,
                point_sets=mesh_model.point_sets,
                cell_sets=mesh_model.cell_sets,
                gmsh_periodic=mesh_model.gmsh_periodic,
                info=mesh_model.info,
        )
        self.point_sets = self.get_point_sets()
        # todo: add cell_sets
        # todo: add field_sets
        self.field_sets = self.point_sets

        pass
    def load_file(self, file):
        r"""
            load mesh file
            :param file:
            :return:
        """
        assert os.path.exists(file), 'file not found: ' + file
        try:
            mesh = meshio.read(file)
            print('meshio.read success!')
        except:
            print('meshio.read failed, try gmsh.open and gmsh.write load mesh!')
            gmsh.initialize()
            gmsh.open(file)
            gmsh.write('temp.msh')
            gmsh.finalize()
            mesh = meshio.read('temp.msh')
            os.remove('temp.msh')
        # self.mesh_model = mesh
        # self.mesh_model = mesh
        return mesh

    def save_file(self, file, file_type='msh'):
        r"""
            save mesh file
            Args:
                :param file: file path
                :param file_type: 'msh', 'vtk', 'xdmf', 'xdmf3', 'stl', 'ply', 'obj', 'off', 'mesh', 'med', 'h5m', 'vtu'
            Return:
                None
        """
        meshio.write(file, self, file_format=file_type)


    def get_point_sets(self, ):
        r"""
            get point sets
            :return: point sets
        """
        point_sets = {}
        for field_name in self.field_data.keys():
            point_index = []
            for cell_type, cell_index in self.cell_sets_dict[field_name].items():
                cell_set = self.cells_dict[cell_type][cell_index]
                point_index.extend(cell_set.flatten())

            points = self.points[np.unique(np.array(point_index, dtype=np.int32)), :]
            # # plot fields data
            # if not os.path.exists('mesh_plots'):
            #     os.mkdir('mesh_plots')
            # fig = plt.figure(1)
            # plt.scatter(points[:, 0], points[:, 1], s=0.5)
            # fig.savefig(os.path.join('mesh_plots', field_name + '.jpg'))

            point_sets.update({field_name: points})
        return point_sets