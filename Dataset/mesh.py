#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/15 1:30
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : MeshLoader.py
# @Description    : ******
"""

import meshio
import os
import gmsh
import pygmsh
class MeshLoader(meshio.Mesh):
    def __init__(self, file):
        mesh_model = self.loadfile(file)
        super(MeshLoader, self).__init__(
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

        pass
    def loadfile(self, file):
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

    def get_point_sets(self, name: str):
        assert name in self.cell_sets_dict.keys(), 'cell set not found: ' + name
        points_sets = []
        for key in self.cell_sets_dict[name]:
            index = self.cell_sets_dict[name][key]
            points_sets.append(self.points[index])
        return points_sets