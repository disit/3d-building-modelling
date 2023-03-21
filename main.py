# DSM Modeler
# Copyright (C) 2022 DISIT Lab https://www.disit.org - University of Florence
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import os
import json
from functions.read_asc import readASC2np
from get_3D_model import get_3D_model, get_3D_model_LOD1

if __name__ == "__main__":

    X_zero = 1681357
    Y_zero = 4848787
    fulldsm = "data/dsm_example.asc"
    fulldtm = "data/dtm_example.asc"
    geojson = "data/building.geojson"
    buildingID = os.path.split(geojson)[1].split('.')[0]
    save_path_glb = "output/" + buildingID + ".glb"
    save_path_glb_lod1 = "output/" + buildingID + "_lod1.glb"
    time_file_path = "output/time.csv"

    # read DSM
    dsm, XLL, YLL, cellsize, nodata_val = readASC2np(fulldsm)
    # read DTM
    dtm, XLL_, YLL_, cellsize_, nodata_val_ = readASC2np(fulldtm)

    with open(geojson, 'r') as f:
        data = json.load(f)
    f.close()

    get_3D_model(data, dsm, dtm, XLL, YLL, cellsize, buildingID, X_zero, Y_zero, save_path_glb, time_file_path)
    get_3D_model_LOD1(data, dsm, dtm, XLL, YLL, cellsize, buildingID, X_zero, Y_zero, save_path_glb_lod1)


