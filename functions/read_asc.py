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
# Author: Marco Fanfani (marco.fanfani@unifi.it)

import numpy as np
from matplotlib import pyplot as plt

def readASC2np(filename):
    #f = open("../../data/Lidar_firenze/dsm/unzipped/19k61_1x1_dsm_011_2007_3003.asc", "r")
    f = open(filename, "r")
    header = {}
    for i in range(5):
        line = str(f.readline()).split()
        header.update({line[0].upper(): float(line[1])})

    ncols = int(header.get("NCOLS"))
    nrows = int(header.get("NROWS"))

    xcenter = header.get("XLLCENTER")
    if xcenter == None:
        xcenter = header.get("XLLCORNER") + 0.5
    ycenter = header.get("YLLCENTER")
    if ycenter == None:
        ycenter = header.get("YLLCORNER") + 0.5

    cellsize = header.get("CELLSIZE")
    nodata_val = header.get("NODATA_VALUE")

    dsm = np.zeros((nrows, ncols), dtype=np.float32)

    for i in range(nrows):
        file_line = str(f.readline()).split()
        dsm[i,:] = np.array(file_line, dtype=np.float32)
        i = i + 1
        if not file_line:
            print("End Of File at row %d\n" % (i))
            break

    
    # plt.figure()
    # plt.imshow(dsm, cmap='gray')
    # plt.show(block=True)

    return dsm, xcenter, ycenter, cellsize, nodata_val
