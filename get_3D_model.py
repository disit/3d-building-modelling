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

import pyransac3d as pyrsc
from skimage.measure import LineModelND, ransac
from skimage.morphology import thin
import math
import cv2
from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.ops import unary_union, polygonize_full
from shapely import affinity
import hdbscan
from functions import canny_edge_detector as ced
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
from skspatial.objects import Line as skLine
import pygltflib
from math import sqrt
import triangle as tr
from stl import mesh
from pyproj import Transformer
import time
import matplotlib
matplotlib.use('Qt5Agg')

def normalize(vector):
    norm = 0
    for i in range(0, len(vector)):
        norm += vector[i] * vector[i]
    norm = sqrt(norm)
    for i in range(0, len(vector)):
        vector[i] = vector[i] / norm

    return vector


def stl2glb(stl_file, glb_file):
    stl_mesh = mesh.Mesh.from_file(stl_file)

    stl_points = []
    stl_tri_idx = []
    for i in range(0, len(stl_mesh.points)):  # Convert points into correct numpy array
        stl_points.append([stl_mesh.points[i][0], stl_mesh.points[i][1], stl_mesh.points[i][2]])
        stl_points.append([stl_mesh.points[i][3], stl_mesh.points[i][4], stl_mesh.points[i][5]])
        stl_points.append([stl_mesh.points[i][6], stl_mesh.points[i][7], stl_mesh.points[i][8]])
        stl_tri_idx.append([i * 3, i * 3 + 1, i * 3 + 2])

    points = np.array(
        stl_points,
        dtype="float32",
    )

    tri_idx = np.array(
        stl_tri_idx,
        dtype="uint16",
    )

    stl_normals = []
    for i in range(0, len(stl_mesh.normals)):  # Convert points into correct numpy array
        normal_vector = [stl_mesh.normals[i][0], stl_mesh.normals[i][1], stl_mesh.normals[i][2]]
        normal_vector = normalize(normal_vector)
        stl_normals.append(normal_vector)
        stl_normals.append(normal_vector)
        stl_normals.append(normal_vector)

    normals = np.array(
        stl_normals,
        dtype="float32"
    )

    points_binary_blob = points.tobytes()
    normals_binary_blob = normals.tobytes()
    tri_idx_binary_blob = tri_idx.flatten().tobytes()

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=0, NORMAL=1),
                        material=0,
                        indices=2
                    )
                ]
            )
        ],
        materials=[
            pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(baseColorFactor=[1.0, 0.0, 0.0, 1.0]),
                doubleSided=True,
                alphaMode=pygltflib.MASK
            )
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.FLOAT,
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(normals),
                type=pygltflib.VEC3,
                max=None,
                min=None,
            ),
            pygltflib.Accessor(
                bufferView=2,
                componentType=pygltflib.UNSIGNED_SHORT,
                count=tri_idx.size,
                type=pygltflib.SCALAR,
                max=None,
                min=None,
            ),
        ],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0,
                byteOffset=0,
                byteLength=len(points_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(points_binary_blob),
                byteLength=len(normals_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(points_binary_blob) + len(normals_binary_blob),
                byteLength=len(tri_idx_binary_blob),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
        ],
        buffers=[
            pygltflib.Buffer(
                byteLength=len(points_binary_blob) + len(normals_binary_blob) + len(tri_idx_binary_blob)
            )
        ],
    )
    gltf.set_binary_blob(points_binary_blob + normals_binary_blob + tri_idx_binary_blob)

    # pbr = pygltflib.PbrMetallicRoughness() # Use PbrMetallicRoughness
    # pbr.baseColorFactor = [1.0, 0.0, 0.0, 1.0] # solid red
    # material = pygltflib.Material()
    # material.pbrMetallicRoughness = pbr
    # material.doubleSided = True # make material double sided
    # material.alphaMode = pygltflib.MASK   # to get around 'MATERIAL_ALPHA_CUTOFF_INVALID_MODE' warning

    # primitive = pygltflib.Primitive()
    # primitive.attributes.POSITION = 0
    # primitive.material = 0

    # gltf.materials.append(material)
    # gltf.meshes[0].primitives.append(primitive)

    gltf.save(glb_file)


def save2glb(points, triangles, glb_file, rgb):
    points = points.tolist()
    triangles = triangles.tolist()

    points = np.array(points, dtype="float32")
    triangles = np.array(triangles, dtype="uint16")

    points_binary_blob = points.tobytes()
    triangles_binary_blob = triangles.flatten().tobytes()

    # https://stackoverflow.com/questions/66469497/gltf-setting-colors-basecolorfactor
    R = math.pow(rgb[0] / 255.0, 2.2)
    G = math.pow(rgb[1] / 255.0, 2.2)
    B = math.pow(rgb[2] / 255.0, 2.2)

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=1),
                        material=0,
                        indices=0
                    )
                ]
            )
        ],
        materials=[
            pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(baseColorFactor=[R, G, B, rgb[3]]),
                doubleSided=True,
                alphaMode=pygltflib.MASK
            )
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.UNSIGNED_SHORT,
                count=triangles.size,
                type=pygltflib.SCALAR,
                max=[int(triangles.max())],
                min=[int(triangles.min())],
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            ),
        ],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0,
                byteLength=len(triangles_binary_blob),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(triangles_binary_blob),
                byteLength=len(points_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
        ],
        buffers=[
            pygltflib.Buffer(
                byteLength=len(triangles_binary_blob) + len(points_binary_blob)
            )
        ],
    )
    gltf.set_binary_blob(triangles_binary_blob + points_binary_blob)
    gltf.save(glb_file)


def coordinates_gis_to_image(X, Y, XLL, YLL, row, inset):
    x = X - (XLL - inset) - 0.5
    y = -(Y - (YLL - inset)) + row - 0.5
    return (x, y)


def coordinates_image_to_gis(x, y, XLL, YLL, row, inset):
    X = x + (XLL - inset) + 0.5
    Y = -y + YLL - inset + row - 0.5
    return (X, Y)


def get_neighbor_idxs(pt, pts, nnum=8):
    x = pt[0]
    y = pt[1]
    if (nnum == 4):
        n = np.zeros((4, 2), dtype=int)
        n[0,] = np.asarray([x, y + 1])
        n[1,] = np.asarray([x, y - 1])
        n[2,] = np.asarray([x - 1, y])
        n[3,] = np.asarray([x + 1, y])
    else:
        n = np.zeros((8, 2), dtype=int)
        n[0,] = np.asarray([x, y + 1])
        n[1,] = np.asarray([x, y - 1])
        n[2,] = np.asarray([x - 1, y])
        n[3,] = np.asarray([x + 1, y])
        n[4,] = np.asarray([x + 1, y + 1])
        n[5,] = np.asarray([x - 1, y + 1])
        n[6,] = np.asarray([x + 1, y - 1])
        n[7,] = np.asarray([x - 1, y - 1])
    idxs = []
    for i in range(np.shape(n)[0]):
        tmp = np.where((pts[:, 0] == n[i, 0]) & (pts[:, 1] == n[i, 1]))[0]
        if (tmp.size == 1):
            idxs.append(tmp[0])
    return idxs


def get_plane_distance(query_pt, plane):
    num = np.abs(np.dot(plane[:3], query_pt) + plane[3])
    den = np.sqrt(np.sum(np.square(plane[:3])))
    return num / den


def get_normal_vectors(dsm_crop, rows, cols):
    normals = np.zeros((rows, cols, 3), dtype=np.float64)
    angles = np.zeros((rows, cols), dtype=np.float64)
    f1 = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]]) / 8
    f2 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]) / 8
    for x in range(1, rows - 1):  # (int x = 0; x < depth.rows; ++x)
        for y in range(1, cols - 1):  # (int y = 0; y < depth.cols; ++y)
            # dzdx = (dsm_crop[x+1, y] - dsm_crop[x-1, y]) / 2.0
            # dzdy = (dsm_crop[x, y+1] - dsm_crop[x, y-1]) / 2.0
            # n = np.array([-dzdx, -dzdy, 1.0])

            t = np.array([x, y - 1, dsm_crop[x - 1, y]])
            l = np.array([x - 1, y, dsm_crop[x, y - 1]])
            c = np.array([x, y, dsm_crop[x, y]])
            n = np.cross((l - c), (t - c))

            # if1 = convolve(dsm_crop, f1)
            # if2 = convolve(dsm_crop, f2)
            # if1 = if1 * -1
            # if2 = if2 * -1
            # sif1 = np.multiply(if1, if1)
            # sif2 = np.multiply(if2, if2)
            # tmp = np.sqrt(sif1 + sif2 + 1)
            # n3 = 1/tmp
            # n1 = if1*n3
            # n2 = if2*n3

            n = n / np.linalg.norm(n)
            angles[x, y] = np.arccos(
                np.clip(np.dot(n, [0, 0, 1]), -1.0, 1.0)) * 180 / np.pi

            n = np.reshape(n, (1, 1, 3))
            normals[x, y, :] = n
    return normals, angles


def get_binary_img(p3d, labels, rows, cols):
    segment = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(np.shape(p3d)[0]):
        x, y = p3d[i, 0:2]
        if (labels[i] == 1):
            segment[int(y), int(x)] = 255
    return segment


def get_clustered_image(p3d, labels, rows, cols, original=False):
    segment = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(np.shape(p3d)[0]):
        if (original):
            lbl = labels[i]
        else:
            lbl = labels[i] + 2
        x, y = p3d[i, 0:2]
        segment[int(y), int(x)] = lbl
    return segment


def show_planes(p3d, polypts_crop, labels, rows, cols):
    segment = get_clustered_image(p3d, labels, rows, cols)
    fig, ax = plt.subplots()
    ax.imshow(segment)
    ax.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
    plt.show(block=False)
    return fig, ax


class Candidate():
    def __init__(self, p1, p2, eq):
        self.p1 = p1
        self.p2 = p2
        self.eq = eq
        self.inl = None
        self.inl_idx = None
        self.sparsity = None

    def point_line_distance(self, pt):
        return self.eq[0] * pt[0] + self.eq[1] * pt[1] + self.eq[2] * pt[2] / np.sqrt(
            self.eq[0] * self.eq[0] + self.eq[1] * self.eq[1])

    def compute_stats(self, pts, th):
        inl = []
        inl_idx = []
        for ip in range(pts.shape[0]):
            if self.eq.distance_point(pts[ip, 0:2]) < th:
                inl.append(pts[ip, 0:2])
                inl_idx.append(ip)
        self.inl = np.array(inl)
        self.inl_idx = np.array(inl_idx)

        # fig, ax = show_planes(data, polypts_crop, np.zeros((data.shape[0])), rows, cols)            
        # ax.plot(self.inl[:,0], self.inl[:,1], 'xr')
        # plt.show(block=False)

        line = skLine.from_points(point_a=self.p1[0:2], point_b=self.p2[0:2])
        proj_pts = []
        for il in range(self.inl.shape[0]):
            proj_pts.append(line.project_point(self.inl[il, :]))
        proj_pts = np.array(proj_pts)
        xs = proj_pts[:, 0]
        ys = proj_pts[:, 1]
        if (xs.shape[0] == np.unique(xs).shape[0]):
            sort_idx = np.argsort(xs)
        elif (ys.shape[0] == np.unique(ys).shape[0]):
            sort_idx = np.argsort(ys)
        else:
            sort_idx = []
            print('doh!')

        max_dist = -1
        if (len(sort_idx) > 0):
            for il in range(self.inl.shape[0] - 1):
                p1 = self.inl[sort_idx[il], :]
                p2 = self.inl[sort_idx[il + 1], :]
                d = np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
                if d > max_dist:
                    max_dist = d
        self.sparsity = max_dist
        # print(max_dist)

        # plt.close(fig)
        return None


def read_geojson_data(data, XLL, YLL, dsm, cellsize, PLOT=False):
    polypts = []
    polypts_img = []
    for f in data['features']:
        npoly = 0
        for ic, coord in enumerate(f['geometry']['coordinates']):
            for i, c in enumerate(coord):
                if isinstance(c[0], list):
                    for i2, c2 in enumerate(c):
                        polypts.append((c2[0], c2[1], ic))
                        polypts_img.append(
                            coordinates_gis_to_image(c2[0], c2[1], XLL, YLL, np.shape(dsm)[0], cellsize / 2)
                        )
                    # npoly += 1
                else:
                    polypts.append((c[0], c[1], ic))
                    polypts_img.append(
                        coordinates_gis_to_image(c[0], c[1], XLL, YLL, np.shape(dsm)[0], cellsize / 2)
                    )

    polypts = np.array(polypts)
    polypts_img = np.asarray(polypts_img)

    upoly = np.unique(polypts[:, 2])
    if upoly.shape[0] > 1:
        areas = []
        for up in upoly:
            vertex = polypts[polypts[:, 2] == up]
            areas.append(Polygon(vertex[:, 0:2]).area)

        main_poly_idx = np.argmax(areas)
        main_polypts = polypts[polypts[:, 2] == upoly[main_poly_idx]]
        main_polypts_img = polypts_img[polypts[:, 2] == upoly[main_poly_idx]]

        sub_polypts = []
        sub_polypts_img = []
        for up in upoly:
            if up != upoly[main_poly_idx]:
                sub_polypts.append(polypts[polypts[:, 2] == up])
                sub_polypts_img.append(polypts_img[polypts[:, 2] == up])
    else:
        main_polypts = polypts
        main_polypts_img = polypts_img
        sub_polypts = []
        sub_polypts_img = []

    if PLOT:
        plt.figure()
        plt.imshow(dsm, cmap='gray')
        plt.plot(main_polypts_img[:, 0], main_polypts_img[:, 1], '-r')
        for i in range(len(sub_polypts_img)):
            plt.plot(sub_polypts_img[i][:, 0], sub_polypts_img[i][:, 1], '-g')
        plt.show(block=False)

    return main_polypts_img, main_polypts, sub_polypts_img, sub_polypts


def region_growing_height_clustering(p3d, polypts_crop, sub_polypts_crop, rows, cols, th_h=0.25, max_iter=100,
                                     PLOT=False):
    seed = p3d[:, 2].argmax()  # start from heigtest point
    height_cluster = []
    label = np.ones((p3d.shape[0]), dtype=int) * -1
    to_test = []
    to_test.append(seed)
    label[seed] = 0
    count = 0
    while not all(label != -1):
        if (len(to_test) > 0):
            idx = to_test.pop()
            h = p3d[idx, 2]
            nidx = get_neighbor_idxs(p3d[idx, :], p3d, nnum=8)
            for i in range(len(nidx)):
                if (abs(h - p3d[nidx[i], 2]) < th_h and label[nidx[i]] == -1):
                    label[nidx[i]] = label[idx]
                    to_test.append(nidx[i])
        else:
            not_evaluated = label == -1
            tmp = p3d[not_evaluated, 2].max()
            new_seed_candidate = np.where(p3d[:, 2] == tmp)[0]
            nsc = 0
            while (True):
                if (label[new_seed_candidate[nsc]] == -1):
                    new_seed = new_seed_candidate[nsc]
                    break
                else:
                    nsc += 1
            # new_seed = np.where(p3d[:, 2] == tmp)[0][0]
            to_test.append(new_seed)
            label[new_seed] = label.max() + 1
        count += 1

    if (PLOT):
        show_planes(p3d, polypts_crop, label, rows, cols)

    ulabel = np.unique(np.asarray(label)).tolist()

    # merge small cluster
    count = 0
    while True:
        ulabel_len = len(ulabel)
        cluster_height = []
        cluster_size = []
        neight_cluster = []
        # for each cluster find its neghtboor clusters
        for i in range(len(ulabel)):
            cpt = p3d[label == ulabel[i], :]
            cluster_height.append(np.median(cpt[:, 2]))
            cluster_size.append(sum(label == ulabel[i]))
            tmp = []
            for p in range(cpt.shape[0]):
                neight = get_neighbor_idxs(cpt[p, :], p3d)
                tmp = [*tmp, *label[neight].tolist()]
            tmp = np.unique(np.array(tmp))
            tmp = np.delete(tmp, tmp == ulabel[i])
            neight_cluster.append(tmp)
        # mergin
        for i in range(len(ulabel)):
            ih = cluster_height[i]
            if sum(label == ulabel[i]) < 10:
                # if i is a small cluster (less than 10 px)                
                found = False
                dmin = 1000
                # search for the neightboor cluster with more similar median height
                for j in neight_cluster[i]:
                    if ulabel[i] != j and cluster_size[ulabel.index(j)] >= 10:
                        # consider only "big" cluster
                        if abs(cluster_height[ulabel.index(j)] - ih) < dmin:
                            dmin = abs(cluster_height[ulabel.index(j)] - ih)
                            close_h = j
                            found = True
                # if found, merge the i-th cluster with the big cluster
                if (found):
                    label[label == ulabel[i]] = close_h
                    tmp = np.hstack(
                        (neight_cluster[i], neight_cluster[ulabel.index(close_h)]))
                    tmp = np.unique(tmp)
                    tmp = np.delete(tmp, tmp == close_h)
                    neight_cluster[ulabel.index(close_h)] = tmp
                    neight_cluster[i] = np.array([])
        if (PLOT):
            show_planes(p3d, polypts_crop, label, rows, cols)

        # continue until no more merging can be done
        ulabel = np.unique(np.asarray(label)).tolist()
        if len(ulabel) == ulabel_len:
            break
        else:
            count += 1
        if count > max_iter:
            raise Exception("Unsuccessful height cluster merging.")
    if PLOT:
        show_planes(p3d, polypts_crop, label, rows, cols)

    new_label = max(ulabel) + 1
    if len(sub_polypts_crop) > 0:
        for sp in range(len(sub_polypts_crop)):
            sp_pt = sub_polypts_crop[sp]
            sp_poly = Polygon(sp_pt)
            for i in range(p3d.shape[0]):
                if Point(p3d[i, 0], p3d[i, 1]).within(sp_poly):
                    label[i] = new_label
            new_label += 1
        ulabel = np.unique(np.asarray(label)).tolist()
    if PLOT:
        show_planes(p3d, polypts_crop, label, rows, cols)

    return label, ulabel


def get_step_lines(p3d, polypts_crop, label, ulabel, rows, cols, small_cluster_th=10, line_inl_th=0.5,
                   min_line_inl_num=2, max_sparsity=3, PLOT=False):
    tmp_idx = np.asarray([n for n in range(p3d.shape[0])])
    step_pt = []
    step_pt_0 = []
    step_pt_idx = []

    for i in range(len(ulabel) - 1):
        for j in range(i + 1, len(ulabel)):

            # ignore small cluster
            if sum(label == ulabel[i]) < small_cluster_th or sum(label == ulabel[j]) < small_cluster_th:
                continue

            # consider the set of points of two cluster i j
            plane_pt_i = p3d[label == ulabel[i], :]
            plane_pt_i_idx = tmp_idx[label == ulabel[i]]
            plane_pt_j = p3d[label == ulabel[j], :]
            plane_pt_j_idx = tmp_idx[label == ulabel[j]]
            plane_pt = np.vstack((plane_pt_i, plane_pt_j))
            plane_pt_idx = np.hstack((plane_pt_i_idx, plane_pt_j_idx))
            step_pt_ij = np.zeros((plane_pt.shape[0]), dtype=bool)

            # for each point in the set find its neighbour and check if some of the found point belong to the other cluster
            for p in range(plane_pt.shape[0]):
                label_p = label[plane_pt_idx[p]]
                neight = get_neighbor_idxs(plane_pt[p, :], plane_pt)
                neight_label = label[plane_pt_idx[neight]]
                neight_label = np.delete(neight_label, neight_label == label_p)
                if neight_label.shape[0] > 0:
                    step_pt_ij[p] = 1

            if (any(step_pt_ij == True)):
                step_pt_0.append(plane_pt[step_pt_ij, :])
                step_pt_ij_copy = np.copy(step_pt_ij)
                if PLOT:
                    show_planes(plane_pt, polypts_crop, step_pt_ij, rows, cols)

                # thinning
                sup_img = get_clustered_image(plane_pt, step_pt_ij, rows, cols, True)
                thinned_img = thin(sup_img)
                ys, xs = np.where((thinned_img == 1))
                step_pt_ij = np.zeros((plane_pt.shape[0]), dtype=bool)
                for n in range(xs.shape[0]):
                    idx = np.where(((plane_pt[:, 0] == xs[n]) & (plane_pt[:, 1] == ys[n])))
                    if (len(idx) == 1 and len(idx[0] == 1)):
                        step_pt_ij[idx[0][0]] = 1

                if PLOT:
                    show_planes(plane_pt, polypts_crop, step_pt_ij, rows, cols)
                if (plane_pt[step_pt_ij, :].shape[0] > 1):
                    step_pt.append(plane_pt[step_pt_ij, :])
                    step_pt_idx.append(step_pt_ij)
                else:
                    step_pt.append(plane_pt[step_pt_ij_copy, :])
                    step_pt_idx.append(step_pt_ij_copy)
    # find lines from step_pt
    step_lines = []
    for i in range(len(step_pt)):

        step_lines_i = []
        line_inliers = []

        data = step_pt[i][:, 0:2]  # step pt between two height clusters

        candidates = []
        inliers = []
        inlier_nums = []
        sparsity = []

        # compute all possible line considering all pairs of pt, and evaluate the inlier set
        for i1 in range(data.shape[0] - 1):
            for i2 in range(i1 + 1, data.shape[0]):
                p1 = data[i1, :]
                p2 = data[i2, :]
                eq = skLine.from_points(point_a=p1[0:2], point_b=p2[0:2])
                cc = Candidate(p1, p2, eq)
                cc.compute_stats(data, line_inl_th)
                candidates.append(cc)
                ss = np.zeros((data.shape[0]), dtype=bool)
                if (cc.inl_idx.shape[0] > 0):
                    ss[cc.inl_idx] = True
                inliers.append(ss)
                inlier_nums.append(ss.sum())
                sparsity.append(cc.sparsity)

        inlier_nums = np.array(inlier_nums)
        sparsity = np.array(sparsity)

        if PLOT:
            fig, ax = show_planes(data, polypts_crop, np.zeros((data.shape[0])), rows, cols)
            for c in range(inlier_nums.shape[0]):
                if (inlier_nums[c] > 0):
                    ax.plot(candidates[c].inl[:, 0], candidates[c].inl[:, 1], 'x', label=str(candidates[c].sparsity))
            ax.legend(loc='lower left')
            plt.show(block=False)
            plt.close(fig)

        # select the best lines
        if (inliers):
            preferences = np.array(inliers)
            goon = True
            while (goon):
                # get the line with most inliers
                inl_num = preferences.sum(axis=1)
                best_num = np.max(inl_num)

                if (best_num > min_line_inl_num):

                    # among lines with the same number of inliers, prefer the one where inliers are close to each other
                    best_idx = np.where((inl_num == best_num))[0]
                    best_spar_idx = np.argmin(sparsity[best_idx])
                    best_idx_ = best_idx[best_spar_idx]
                    if (sparsity[best_idx_] < max_sparsity):

                        # compute the line equation
                        model = LineModelND()
                        model.estimate(candidates[best_idx_].inl)
                        vm = model.params[1]

                        # evaluate the angles among other already selected lines
                        step_angles = []
                        for s in range(len(step_lines_i)):
                            vs = step_lines_i[s].params[1]
                            step_angles.append(np.arccos(np.clip(np.dot(vm, vs), -1.0, 1.0)) * 180 / np.pi)

                        # accept the new line, only if it is almost perpendicualr o parallel to other already selected lines
                        # >> this constraint is based on the hypotyhesis that if two height-clusters have a step border composed by
                        # >> - a single line: then all candidate should be parallels
                        # >> - multiple lines: then the candidate line should be arranged in a rectangular shape
                        add_line = True
                        for s in range(len(step_angles)):
                            if (10 < step_angles[s] and step_angles[s] < 80) or (
                                    100 < step_angles[s] and step_angles[s] < 170):
                                add_line = False
                        if (add_line):
                            step_lines_i.append(model)
                            line_inliers.append(candidates[best_idx_].inl)

                    # be the line added or not, remove its consensus
                    preferences[best_idx_, :] = False

                else:
                    goon = False

            # remove similar lines, by first grouping lines in similar position and then estimate a single line for each group
            line_groups = []
            group_reps = []
            poly = Polygon(polypts_crop)
            for i in range(len(step_lines_i)):
                l = skLine(step_lines_i[i].params[0], step_lines_i[i].params[1])
                new_rep = True
                for ig, g in enumerate(group_reps):
                    if all(g.point == l.point) and all(g.direction == l.direction):
                        # g and l are the same identical line
                        new_rep = False
                        line_groups[ig] = [*line_groups[ig], i]
                    else:
                        try:
                            pt = g.intersect_line(l)
                        except:
                            pt = []
                        if len(pt) > 0:
                            if (poly.contains(Point(pt[0], pt[1])) == True):
                                angle = np.arccos(np.clip(np.dot(g.direction, l.direction), -1.0, 1.0)) * 180 / np.pi
                                if (0 <= angle and angle < 10) or (170 < angle and angle <= 180):
                                    new_rep = False
                                    line_groups[ig] = [*line_groups[ig], i]
                if new_rep:
                    line_groups.append([i])
                    group_reps.append(l)

            step_lines_i = []
            for g in line_groups:
                g_inl = np.empty((0, 2))
                for l_idx in g:
                    g_inl = np.vstack((g_inl, line_inliers[l_idx]))
                g_inl = np.unique(g_inl, axis=0)
                model = LineModelND()
                model.estimate(g_inl)
                step_lines_i.append(model)

            step_lines = [*step_lines, *step_lines_i]

    # plot estimated step lines
    if PLOT:
        fig, ax = show_planes(p3d, polypts_crop, label, rows, cols)
        for mr in step_lines:
            try:
                line_x = np.arange(p3d[:, 0].min(), p3d[:, 0].max())
                line_y = mr.predict_y(line_x)
                ax.plot(line_x, line_y, '-b')
            except:
                line_y = np.arange(p3d[:, 1].min(), p3d[:, 1].max())
                line_x = mr.predict_x(line_y)
                ax.plot(line_x, line_y, '-b')
        plt.grid('on')
        plt.show(block=False)

    return step_lines


def get_inner_planes(p3d, polypts_crop, rows, cols, nor, grd, drc, ransac_th=0.05, ransac_max_iter=10000,
                     plane_dist_th=0.5, max_expansion_iter=100, PLOT=False):
    # compute distance metrics
    mdist = np.zeros((np.shape(p3d)[0], np.shape(p3d)[0]))
    hdiff = np.zeros((np.shape(p3d)[0], np.shape(p3d)[0]))
    ndiff = np.zeros((np.shape(p3d)[0], np.shape(p3d)[0]))
    gdiff = np.zeros((np.shape(p3d)[0], np.shape(p3d)[0]))
    ddiff = np.zeros((np.shape(p3d)[0], np.shape(p3d)[0]))
    for i in range(np.shape(p3d)[0] - 1):
        for j in range(i + 1, np.shape(p3d)[0]):
            if i != j:
                pi = p3d[i, :]
                pj = p3d[j, :]
                dx = np.abs(pi[0] - pj[0])
                dy = np.abs(pi[1] - pj[1])
                if dx <= 3 and dy <= 3:
                    ni = nor[i, :]
                    nj = nor[j, :]
                    na = np.arccos(
                        np.clip(np.dot(ni, nj), -1.0, 1.0)) * 180 / np.pi
                    hd = abs(pi[2] - pj[2])
                    gd = abs(grd[i] - grd[j])
                    dd = abs(drc[i] - drc[j])
                    mdist[i, j] = dx + dy
                    ndiff[i, j] = na
                    hdiff[i, j] = hd
                    gdiff[i, j] = gd
                    ddiff[i, j] = dd
                    mdist[j, i] = dx + dy
                    ndiff[j, i] = na
                    hdiff[j, i] = hd
                    gdiff[j, i] = gd
                    ddiff[j, i] = dd
                else:
                    mdist[i, j] = np.inf
                    hdiff[i, j] = np.inf
                    ndiff[i, j] = np.inf
                    gdiff[i, j] = np.inf
                    ddiff[i, j] = np.inf
                    mdist[j, i] = np.inf
                    hdiff[j, i] = np.inf
                    ndiff[j, i] = np.inf
                    gdiff[j, i] = np.inf
                    ddiff[j, i] = np.inf

    def normalizer(x):
        max = np.where(np.isinf(x), -np.Inf, x).max()
        min = x.min()
        return (x - min) / (max - min)

    mdist = normalizer(mdist)
    hdiff = normalizer(hdiff)
    ndiff = normalizer(ndiff)
    gdiff = normalizer(gdiff)
    ddiff = normalizer(ddiff)

    weights = mdist + 10 * ndiff + 0.1 * gdiff + 5 * ddiff
    # weights = ndiff
    clusterer = hdbscan.HDBSCAN(metric='precomputed')
    clusterer.fit(weights)
    labels = clusterer.labels_

    if PLOT:
        show_planes(p3d, polypts_crop, labels, rows, cols)

    if (all(labels == -1)):
        labels = np.zeros((labels.shape[0]), dtype=labels.dtype)
        pt_in_plane = np.zeros((np.shape(p3d)[0]), dtype=int)
    else:
        # inner plane refinement with RANSAC
        alabels = np.asarray(labels)
        unique_labels = np.unique(alabels).tolist()
        main_labels = []
        th_min_num_px = 10
        planes = []
        pt_in_plane = np.ones((np.shape(p3d)[0]), dtype=int) * -1
        count = 0
        for l in unique_labels:
            if l != -1:
                tmp = alabels[alabels == l]
                if len((tmp.tolist())) > th_min_num_px:
                    main_labels.append(l)
                    plane_pt = p3d[labels == l, :]
                    idx_pt = np.array(range(np.shape(p3d)[0]))[labels == l]
                    plane = pyrsc.Plane()
                    best_eq, best_inliers = plane.fit(plane_pt, ransac_th, maxIteration=ransac_max_iter)
                    inl_idx = idx_pt[best_inliers]
                    pt_in_plane[inl_idx] = count
                    planes.append(best_eq)
                    count += 1
        if PLOT:
            show_planes(p3d, polypts_crop, pt_in_plane, rows, cols)

        if (len(planes) == 0):
            labels = np.zeros((labels.shape[0]), dtype=labels.dtype)
            pt_in_plane = np.zeros((np.shape(p3d)[0]), dtype=int)
        else:
            # inner plane expansion based on point to plane distance
            while True:
                tmp = np.copy(pt_in_plane)
                for i in range(np.shape(p3d)[0]):
                    if pt_in_plane[i] == -1:
                        pt = p3d[i, :]
                        nidx = get_neighbor_idxs(pt, p3d)
                        near_planes = np.unique(pt_in_plane[nidx])
                        near_planes = np.delete(near_planes, near_planes == -1)
                        dist = []
                        for n in near_planes:
                            if (n != -1):
                                plane_eq = planes[n]
                                dist.append(get_plane_distance(pt, plane_eq))

                        # if the non-associated point has more than one neightbour plane, get the closest one;
                        # otherwise, if it has a single neighbour plane, associate to it only if its distance is less than a threshold
                        # The idea is to avoid that non-asociated points would be associated to the wrong plane only becase it is its single neighbour. 
                        # By postponing the decision, on future iterations, other (maybe closest) planes could reach the non-associated point
                        if len(dist) >= 2 or (len(dist) == 1 and dist[0] < plane_dist_th):
                            closest_plane_idx = dist.index(min(dist))
                            tmp[i] = near_planes[closest_plane_idx]

                if all((pt_in_plane - tmp) == 0):
                    break
                else:
                    pt_in_plane = np.copy(tmp)
                if PLOT:
                    show_planes(p3d, polypts_crop, pt_in_plane, rows, cols)

            pt_in_plane = np.copy(tmp)
            if PLOT:
                show_planes(p3d, polypts_crop, pt_in_plane, rows, cols)

            # associate remaining isolated points
            count = 0
            while True:
                tmp = np.copy(pt_in_plane)
                for i in range(np.shape(p3d)[0]):
                    if pt_in_plane[i] == -1:
                        pt = p3d[i, :]
                        nidx = get_neighbor_idxs(pt, p3d)
                        near_planes = np.unique(pt_in_plane[nidx])
                        near_planes = np.delete(near_planes, near_planes == -1)
                        dist = []
                        for n in near_planes:
                            if (n != -1):
                                plane_eq = planes[n]
                                dist.append(get_plane_distance(pt, plane_eq))
                        if dist:
                            closest_plane_idx = dist.index(min(dist))
                            tmp[i] = near_planes[closest_plane_idx]
                if all((pt_in_plane - tmp) == 0):
                    break
                else:
                    pt_in_plane = np.copy(tmp)
                    count += 1
                if count > max_expansion_iter:
                    raise Exception("Unsuccessful inner plane cluster merging.")
            pt_in_plane = np.copy(tmp)

    if PLOT:
        show_planes(p3d, polypts_crop, pt_in_plane, rows, cols)

    return pt_in_plane


def get_internal_lines(p3d, polypts_crop, pt_in_plane, inner_lines, rows, cols, ransac_th=1, ransac_max_iter=5000,
                       small_cluster_th=10, PLOT=False):
    upt_in_plane = np.unique(pt_in_plane).tolist()
    tmp_idx = np.asarray([n for n in range(p3d.shape[0])])
    inner_pt = []
    for i in range(len(upt_in_plane) - 1):
        for j in range(i + 1, len(upt_in_plane)):
            if sum(pt_in_plane == upt_in_plane[i]) < small_cluster_th or sum(
                    pt_in_plane == upt_in_plane[j]) < small_cluster_th:
                continue
            plane_pt_i = p3d[pt_in_plane == upt_in_plane[i], :]
            plane_pt_i_idx = tmp_idx[pt_in_plane == upt_in_plane[i]]
            plane_pt_j = p3d[pt_in_plane == upt_in_plane[j], :]
            plane_pt_j_idx = tmp_idx[pt_in_plane == upt_in_plane[j]]
            plane_pt = np.vstack((plane_pt_i, plane_pt_j))
            plane_pt_idx = np.hstack((plane_pt_i_idx, plane_pt_j_idx))
            inner_pt_ij = np.zeros((plane_pt.shape[0]), dtype=bool)
            for p in range(plane_pt.shape[0]):
                label_p = pt_in_plane[plane_pt_idx[p]]
                neight = get_neighbor_idxs(plane_pt[p, :], plane_pt)
                neight_label = pt_in_plane[plane_pt_idx[neight]]
                neight_label = np.delete(neight_label, neight_label == label_p)
                if neight_label.shape[0] > 0:
                    inner_pt_ij[p] = 1
            if (any(inner_pt_ij == True)):
                if PLOT:
                    show_planes(plane_pt, polypts_crop, inner_pt_ij, rows, cols)
                # thinning
                sup_img = get_clustered_image(plane_pt, inner_pt_ij, rows, cols, True)
                thinned = thin(sup_img)
                ys, xs = np.where((thinned == 1))
                step_pt_ij = np.zeros((plane_pt.shape[0]), dtype=bool)
                for n in range(xs.shape[0]):
                    idx = np.where(((plane_pt[:, 0] == xs[n]) & (plane_pt[:, 1] == ys[n])))
                    if (len(idx) == 1 and len(idx[0] == 1)):
                        step_pt_ij[idx[0][0]] = 1
                if PLOT:
                    show_planes(plane_pt, polypts_crop, step_pt_ij, rows, cols)
                if (step_pt_ij.sum() >= 2):
                    inner_pt.append(plane_pt[step_pt_ij, :])

    # get lines from internal border points
    for i in range(len(inner_pt)):

        data = inner_pt[i][:, 0:2]

        # # fit line using all data
        # model = LineModelND()
        # model.estimate(data)

        # robustly fit line only using inlier data with RANSAC algorithm
        model_robust, inliers = ransac(data, LineModelND, min_samples=2, residual_threshold=ransac_th,
                                       max_trials=ransac_max_iter)
        outliers = inliers == False

        inner_lines.append(model_robust)

        if PLOT:
            # generate coordinates of estimated models
            test_x = np.arange(data[:, 0].min(), data[:, 0].max())
            test_y = np.arange(data[:, 1].min(), data[:, 1].max())
            try:
                line_x_robust = test_x
                line_y_robust = model_robust.predict_y(test_x)
            except:
                line_x_robust = model_robust.predict_x(test_y)
                line_y_robust = test_y

            fig, ax = show_planes(p3d, polypts_crop, pt_in_plane, rows, cols)
            ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6, label='Inlier data')
            ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6, label='Outlier data')
            ax.plot(line_x_robust, line_y_robust, '-b', label='Robust line model')
            ax.legend(loc='lower left')
            plt.show(block=False)

    return inner_lines


def extract_line_segments(lines, poly, dsm_crop, polypts_crop, sub_polypts_img, PLOT=False):
    lring = LinearRing(list(poly.exterior.coords))
    edges = []
    edges_label = []

    # create LineString objects from the estimated lines computing their intersections with the footprint polygon
    if PLOT:
        fig1, ax1 = plt.subplots()
        ax1.imshow(dsm_crop, cmap='gray')
        ax1.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')

    for i in range(len(lines)):
        xs = [-100, 100]
        try:
            ys = lines[i].predict_y(xs)
        except:
            pt = lines[i].params[0]
            dir = abs(lines[i].params[1])
            if (-0.0001 < dir[0] and dir[0] < 0.0001 and -0.0001 < dir[1] - 1 and dir[1] - 1 < 0.0001):
                ys = [pt[1], pt[1]]
            else:
                continue
        ls = LineString([(xs[0], ys[0]), (xs[1], ys[1])])
        ls = affinity.scale(ls, xfact=100, yfact=100)
        it = lring.intersection(ls)
        if it.geom_type == "LineString" or it.geom_type == "Point":
            continue
        pt = list(it)
        pt_list = []
        for p in pt:
            if p.geom_type == "Point":
                pt_list.append((p.x, p.y))
        if len(pt_list) < 2:
            continue
        first, last = LineString(pt_list).boundary
        edges.append(LineString([first, last]))
        edges_label.append(0)

        xs = [first.x, last.x]
        ys = [first.y, last.y]

        if PLOT:
            ax1.plot(xs, ys, 'x-g')
            plt.show(block=False)

    # create LineString objects from subsequent vertex of the footprint polygon
    bpt = np.vstack((lring.xy[0], lring.xy[1])).T
    for i in range(bpt.shape[0] - 1):
        if PLOT:
            xs = [bpt[i, 0], bpt[i + 1, 0]]
            ys = [bpt[i, 1], bpt[i + 1, 1]]
            ax1.plot(xs, ys, 'x-g')
            plt.show(block=False)

        tmp = LineString([bpt[i, :], bpt[i + 1, :]])
        tmp = affinity.scale(tmp, xfact=100, yfact=100)
        it = lring.intersection(tmp)
        if it.geom_type == "GeometryCollection":
            lpts = []
            for el in it:
                if el.geom_type == 'LineString':
                    first, last = el.boundary
                    lpts.append([first.x, first.y])
                    lpts.append([last.x, last.y])
                elif el.geom_type == 'Point':
                    lpts.append([el.x, el.y])
            lpts_ = np.array(lpts)
            deltax = lpts_[:, 0].max() - lpts_[:, 0].min()
            deltay = lpts_[:, 1].max() - lpts_[:, 1].min()
            if deltax >= deltay:
                lpts_ = lpts_[lpts_[:, 0].argsort(), :]
            else:
                lpts_ = lpts_[lpts_[:, 1].argsort(), :]
            it = LineString(lpts_)
        first, last = it.boundary
        edges.append(LineString([first, last]))
        edges_label.append(1)
        # print(pt)    
        xs = [first.x, last.x]
        ys = [first.y, last.y]
        if PLOT:
            ax1.plot(xs, ys, 'x-b')
            plt.show(block=False)

    # if there are inner polygons (i.e. the input shape is a multipolygon), add also thier line segments
    if len(sub_polypts_img) > 0:
        for sp in range(len(sub_polypts_img)):
            bpt = sub_polypts_img[sp]
            for i in range(bpt.shape[0] - 1):
                if PLOT:
                    xs = [bpt[i, 0], bpt[i + 1, 0]]
                    ys = [bpt[i, 1], bpt[i + 1, 1]]
                    ax1.plot(xs, ys, 'x-g')
                    plt.show(block=False)

                tmp = LineString([bpt[i, :], bpt[i + 1, :]])
                tmp = affinity.scale(tmp, xfact=100, yfact=100)
                it = lring.intersection(tmp)
                if it.geom_type == "GeometryCollection" or it.geom_type == "MultiPoint":
                    lpts = []
                    for el in it:
                        if el.geom_type == 'LineString':
                            first, last = el.boundary
                            lpts.append([first.x, first.y])
                            lpts.append([last.x, last.y])
                        elif el.geom_type == 'Point':
                            lpts.append([el.x, el.y])
                    lpts_ = np.array(lpts)
                    deltax = lpts_[:, 0].max() - lpts_[:, 0].min()
                    deltay = lpts_[:, 1].max() - lpts_[:, 1].min()
                    if deltax >= deltay:
                        lpts_ = lpts_[lpts_[:, 0].argsort(), :]
                    else:
                        lpts_ = lpts_[lpts_[:, 1].argsort(), :]
                    it = LineString(lpts_)
                first, last = it.boundary
                edges.append(LineString([first, last]))
                edges_label.append(1)
                # print(pt)    
                xs = [first.x, last.x]
                ys = [first.y, last.y]
                if PLOT:
                    ax1.plot(xs, ys, 'x-m')
                    plt.show(block=False)

    return edges, edges_label


def filter_lines(edges, edges_label, poly, rows, cols, dsm_crop, polypts_crop, ransac_th=1, ransac_max_iter=5000,
                 PLOT=False):
    consensus = np.zeros((len(edges), (rows * cols)), dtype=bool)
    for i in range(len(edges)):
        li = edges[i]
        p0i = np.array([li.xy[0][0], li.xy[1][0], 1])
        p1i = np.array([li.xy[0][1], li.xy[1][1], 1])
        eqi = np.cross(p0i, p1i)
        newpti = []
        for xnew in range(cols):  # np.arange(p0i[0],p1i[0]):
            if eqi[1] != 0:
                ynew = round((-eqi[0] * xnew - eqi[2]) / eqi[1])
                if 0 <= ynew and ynew < rows:
                    newpti.append([xnew, ynew])
        for ynew in range(rows):  # np.arange(p0i[0],p1i[0]):
            if eqi[0] != 0:
                xnew = round((-eqi[1] * ynew - eqi[2]) / eqi[0])
                if 0 <= xnew and xnew < cols:
                    newpti.append([xnew, ynew])
        newpti = np.array(newpti)
        support = np.zeros((rows, cols), dtype=bool)
        support[newpti[:, 1], newpti[:, 0]] = True
        consensus[i, :] = support.flatten()
        # support_ = np.reshape(consensus[i,:], (rows, cols))
        # print((~np.bitwise_xor(support, support_)).all())

        # if PLOT:
        # fig1, ax1 = plt.subplots()
        # ax1.imshow(support, cmap='gray')
        # ax1.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
        # ax1.plot(edges[i].xy[0], edges[i].xy[1], 'b-x')
        # ax1.plot([p0i[0],p1i[0]], [p0i[1],p1i[1]], 'b-o')
        # plt.show(block=True)

        # dist2 = np.abs(eqi[0]*c + eqi[1]*r + eqi[2])/np.sqrt( eqi[0]*eqi[0] + eqi[1]*eqi[1]  )
    group = []
    group_idxs = []
    for i in range(consensus.shape[0]):
        ci = consensus[i, :]
        new_group = True
        similarity = []
        for g in group:
            intersection = ci & g
            union = ci | g
            similarity.append(sum(intersection) / sum(union))
        if (similarity):
            max_similarity = max(similarity)
            if (max_similarity > 0.1):
                new_group = False
                sim_idx = similarity.index(max_similarity)
                group[sim_idx] = group[sim_idx] | ci
                group_idxs[sim_idx] = [*group_idxs[sim_idx], i]
        if new_group:
            group.append(ci)
            group_idxs.append([i])

    line_to_keep = []
    for g in group_idxs:
        if len(g) > 1:
            tmp = [edges_label[ig] for ig in g]
            if (np.array(tmp) == 1).any():
                for l in g:
                    if edges_label[l] == 1:
                        line_to_keep.append(edges[l])
            else:  # TODO: CHECK THIS PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                pt_set = []
                for l in g:
                    cl = np.reshape(consensus[l, :], (rows, cols))
                    pt = np.column_stack(np.where(cl))
                    if len(pt_set) > 0:
                        pt_set = np.vstack((pt_set, pt))
                    else:
                        pt_set = pt

                pt_set = np.unique(pt_set, axis=0)
                pt_set[:, [1, 0]] = pt_set[:, [0, 1]]  # swap columns to move from yx to xy
                model_robust, inliers = ransac(pt_set, LineModelND, min_samples=2, residual_threshold=ransac_th,
                                               max_trials=ransac_max_iter)

                # FROM HERE it should be moved to function (same code in "extract_line_segments")
                xs = [-100, 100]
                lring = LinearRing(list(poly.exterior.coords))
                try:
                    ys = model_robust.predict_y(xs)
                except:
                    pt = model_robust.params[0]
                    dir = model_robust.params[1]
                    if (-0.0001 < dir[0] and dir[0] < 0.0001 and -0.0001 < dir[1] - 1 and dir[1] - 1 < 0.0001):
                        ys = [pt[1], pt[1]]
                ls = LineString([(xs[0], ys[0]), (xs[1], ys[1])])
                it = lring.intersection(ls)
                if it.geom_type != "LineString" and it.geom_type != "Point":
                    pt = list(it)
                    pt_list = []
                    for p in pt:
                        if p.geom_type == "Point":
                            pt_list.append((p.x, p.y))
                    if len(pt_list) >= 2:
                        first, last = LineString(pt_list).boundary
                        line_to_keep.append(LineString([first, last]))
                # TO HERE
        else:
            line_to_keep.append(edges[g[0]])
        if PLOT:
            fig1, ax1 = plt.subplots()
            ax1.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
            for l in g:
                ax1.plot(edges[l].xy[0], edges[l].xy[1], '-x')
            plt.show(block=False)

    return line_to_keep

    # remove similar lines ####################################################################
    # def get_angle(ref_line,card_line):
    #     mr=-(ref_line[0]/ref_line[1])
    #     mc=-(card_line[0]/card_line[1])
    #     ar=np.arctan(mr)
    #     ac=np.arctan(mc)

    #     return abs(ar-ac)*180/np.pi

    # line_sim = np.zeros((len(edges), len(edges)), dtype=float)
    # for i in range(len(edges)-1):
    #     for j in range(i+1,len(edges)):
    #         li = edges[i]
    #         p0i = np.array([li.xy[0][0], li.xy[1][0], 1])
    #         p1i = np.array([li.xy[0][1], li.xy[1][1], 1])
    #         eqi = np.cross(p0i,p1i)
    #         lj = edges[j]
    #         p0j = np.array([lj.xy[0][0], lj.xy[1][0], 1])
    #         p1j = np.array([lj.xy[0][1], lj.xy[1][1], 1])
    #         eqj = np.cross(p0j,p1j)
    #         # fig1, ax1 = plt.subplots()
    #         # ax1.imshow(dsm_crop, cmap='gray')
    #         # ax1.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
    #         # ax1.plot(edges[i].xy[0], edges[i].xy[1], 'b-x')
    #         # ax1.plot([p0i[0],p1i[0]], [p0i[1],p1i[1]], 'b-o')
    #         # newpti = []
    #         # for xnew in np.arange(p0i[0],p1i[0]):
    #         #     ynew = (-eqi[0]*xnew -eqi[2])/eqi[1]
    #         #     newpti.append([xnew, ynew])
    #         # newpti = np.array(newpti)
    #         # ax1.plot(newpti[:,0], newpti[:,1], 'b--+')
    #         # ax1.plot(edges[j].xy[0], edges[j].xy[1], 'g-x')
    #         # ax1.plot([p0j[0],p1j[0]], [p0j[1],p1j[1]], 'g-o')
    #         # plt.show(block=False)

    #         # fig1, ax1 = plt.subplots()
    #         # ax1.imshow(dsm_crop, cmap='gray')
    #         # ax1.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
    #         # ax1.plot(edges[i].xy[0], edges[i].xy[1], 'b-x')
    #         # ax1.plot(edges[j].xy[0], edges[j].xy[1], 'g-x')
    #         mid_i = [np.mean(edges[i].xy[0]), np.mean(edges[i].xy[1])]
    #         mid_j = [np.mean(edges[j].xy[0]), np.mean(edges[j].xy[1])]
    #         # ax1.plot(mid_i[0], mid_i[1], 'or')
    #         # ax1.plot(mid_j[0], mid_j[1], 'og')
    #         # plt.show(block=False)

    #         mid_dist = np.sqrt( (mid_i[0]-mid_j[0])*(mid_i[0]-mid_j[0]) + (mid_i[1]-mid_j[1])*(mid_i[1]-mid_j[1])  )

    #         li_ = affinity.scale(li, xfact=1.5, yfact=1.5)
    #         lj_ = affinity.scale(lj, xfact=1.5, yfact=1.5)
    #         if (li_.intersection(lj_)).is_empty or mid_dist > 3:
    #             angle_ij = np.inf
    #         else:
    #             angle_ij = get_angle(eqi,eqj)
    #         line_sim[i,j] = angle_ij
    #         line_sim[j,i] = line_sim[i,j]

    # edges_to_remove = np.zeros((len(edges)), dtype=bool)
    # for i in range(len(edges)-1):
    #     for j in range(i+1,len(edges)):
    #         # if (-1 < line_sim[i,j] and line_sim[i,j] < -0.9) or (0.9 < line_sim[i,j] and line_sim[i,j] < 1):
    #         if (-15 < line_sim[i,j] and line_sim[i,j] < 15):
    #             if(edges_label[i]==0 and edges_label[j]==1):
    #                 edges_to_remove[i] = True
    #             elif(edges_label[j]==0 and edges_label[i]==1):
    #                 edges_to_remove[j] = True
    #             print(line_sim[i,j])
    #             # fig1, ax1 = plt.subplots()
    #             # ax1.imshow(dsm_crop, cmap='gray')
    #             # ax1.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
    #             # ax1.plot(edges[i].xy[0], edges[i].xy[1], 'b-x')
    #             # ax1.plot(edges[j].xy[0], edges[j].xy[1], 'g-x')
    #             # plt.show(block=False)
    # edges = [edges[i] for i in range(edges_to_remove.shape[0]) if not edges_to_remove[i]]

    # fig1, ax1 = plt.subplots()
    # ax1.imshow(dsm_crop, cmap='gray')
    # ax1.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
    # for i in range(len(edges)):
    #     ax1.plot(edges[i].xy[0], edges[i].xy[1], 'b-x')
    # plt.show(block=False)


def get_polygon_patches(edges, p3d_all, polypts_crop, label, rows, cols, PLOT=False):
    # split the created LineSegment by computing their intersections

    if PLOT:
        fig1, ax1 = show_planes(p3d_all, polypts_crop, label, rows, cols)

    edges2 = []
    for i in range(len(edges)):
        intersections = []
        tmp_i = affinity.scale(edges[i], xfact=100,
                               yfact=100)  # extend the line segments to guarantee to find all intersections
        for j in range(len(edges)):
            if i == j:
                continue
            tmp_j = affinity.scale(edges[j], xfact=100, yfact=100)
            pt = tmp_i.intersection(tmp_j)
            if (not pt.is_empty and pt.geom_type == 'Point'):
                x, y = pt.xy
                intersections.append((x[0], y[0]))
        intersections = np.array(intersections)

        # remove duplicated intersections
        intersections = intersections[intersections[:, 0].argsort(), :]
        intersections = np.unique(intersections, axis=0)
        tmp = intersections[0:-2, :] - intersections[1:-1, :]
        tmp = np.where(((abs(tmp[:, 0]) < 0.01) & (abs(tmp[:, 1]) < 0.01)))
        intersections = np.delete(intersections, tmp, axis=0)

        # create LineString object for each splitted line segment
        for k in range(intersections.shape[0] - 1):
            edges2.append(LineString(
                [(intersections[k, 0], intersections[k, 1]), (intersections[k + 1, 0], intersections[k + 1, 1])]))
            if PLOT:
                ax1.plot([intersections[k, 0], intersections[k + 1, 0]], [intersections[k, 1], intersections[k + 1, 1]],
                         '-g')
                ax1.plot([intersections[k, 0], intersections[k + 1, 0]], [intersections[k, 1], intersections[k + 1, 1]],
                         'xm')
                plt.show(block=False)

    # compute the polygon patches defined by the splitted lines
    # edges2.append(poly.boundary)
    edges2 = unary_union(edges2)
    # edges2 = linemerge(edges2)
    # polygons = list(polygonize(edges2))
    result, dangles, cuts, invalids = polygonize_full(edges2)
    polygons = list(result.geoms)

    to_keep = []
    for i in range(len(polygons)):
        x, y = polygons[i].exterior.xy
        if not (any(np.array(x) < 0) or any(np.array(x) > cols) or any(np.array(y) < 0) or any(np.array(y) > rows)):
            to_keep.append(polygons[i])
        a = 0
    polygons = to_keep

    if PLOT:
        tmp = np.arange(len(polygons))
        colors = plt.get_cmap("tab20")(tmp / (tmp.max() if tmp.max() > 0 else 1))
        fig, ax = show_planes(p3d_all, polypts_crop, label, rows, cols)
        for i in range(len(polygons)):
            x, y = polygons[i].exterior.xy
            plt.fill(x, y, color=colors[i, :])
            plt.plot(x, y, '-m')
        plt.show(block=False)

    return polygons


def get_label_superset(poly, p3d_all, pt_in_plane_all, label, delta=0.5, step=0.1):
    # create a denser point set by super-sampling the dsm pixels. Each super-sample get the plane label of the original pixel

    p3d_superset = []
    pt_in_plane_superset = []
    pt_height_superset = []
    for k in range(p3d_all.shape[0]):
        p = p3d_all[k, 0:2]
        z = p3d_all[k, 2]
        l = pt_in_plane_all[k]  # label[k] #
        l2 = label[k]
        for i in np.arange(p[0] - delta, p[0] + delta, step):
            for j in np.arange(p[1] - delta, p[1] + delta, step):
                if (poly.contains(Point(i, j)) == True):
                    p3d_superset.append([i, j, z])
                    pt_in_plane_superset.append(l)
                    pt_height_superset.append(l2)
    p3d_superset = np.asarray(p3d_superset)
    pt_in_plane_superset = np.asarray(pt_in_plane_superset)
    pt_height_superset = np.array(pt_height_superset)

    return p3d_superset, pt_in_plane_superset, pt_height_superset


def get_polygon_patch_label(polygons, p3d_superset, pt_height_superset, pt_in_plane_superset, p3d_all, polypts_crop,
                            pt_in_plane_all, rows, cols, PLOT=False):
    # assign a label to each polygon patch
    poly_label = np.ones((len(polygons))) * -1
    for it, t in enumerate(polygons):
        x, y = t.exterior.xy
        vertexes = np.vstack((np.array(list(x)), np.array(list(y)))).T
        p = Path(vertexes)
        into = p.contains_points(p3d_superset[:, 0:2])  # ,  radius=-1.0)
        if (any(into)):
            into_height = pt_height_superset[into]  # pt_in_plane_superset[into]
            main_height = np.argmax(np.bincount(into_height))
            subidx = np.where(((pt_height_superset == main_height) & (into == True)))[0]
            into_plane = pt_in_plane_superset[subidx]
            if (np.unique(into_plane).shape[0] > 1):
                a = 0
            # if any(into_plane==-1):
            #    continue
            # main_label = np.argmax(np.bincount(into_plane))
            values, counts = np.unique(into_plane, return_counts=True)
            main_label = values[np.argmax(counts)]
            poly_label[it] = main_label

    u_poly_label = np.unique(poly_label)

    if PLOT:
        tmp = np.arange(u_poly_label.shape[0])
        colors = plt.get_cmap("tab20")(tmp / (tmp.max() if tmp.max() > 0 else 1))
        patches = []
        from matplotlib.patches import Polygon as pltPoly
        for it, t in enumerate(polygons):
            idx = np.where(u_poly_label == poly_label[it])[0][0]
            x, y = t.exterior.xy
            vertexes = np.vstack((np.array(list(x)), np.array(list(y)))).T
            patches.append(pltPoly(vertexes, closed=True, color=colors[idx]))

        from matplotlib.collections import PatchCollection
        fig, ax = show_planes(p3d_all, polypts_crop, pt_in_plane_all, rows, cols)
        p = PatchCollection(patches, alpha=1, match_original=True)
        ax.add_collection(p)
        ax.autoscale()
        plt.show(block=False)

    return poly_label, u_poly_label


def merge_polygon_patches(polygons, poly_label, u_poly_label):
    merged = []
    merged_lb = []
    for ll in u_poly_label:
        if (ll == -1):
            continue
        to_merge = []
        for ip, pp in enumerate(polygons):
            if (pp.boundary.geom_type == 'MultiLineString'):
                print('Error at polygon ' + str(ip))
            if (poly_label[ip] == ll):
                to_merge.append(pp)
        union = unary_union(to_merge)
        if union.geom_type == 'MultiPolygon' or union.boundary.geom_type == 'MultiPoint' or union.boundary.geom_type == 'MultiLineString':
            for pp in to_merge:
                merged.append(pp)
                merged_lb.append(ll)
        else:
            merged.append(union)
            merged_lb.append(ll)

    return merged, merged_lb


def get_3D_model_pts(p3d_all, pt_in_plane_all, pt_in_plane_supp, merged, merged_lb, ground_height, dsm_crop,
                     polypts_crop, rows, cols, ransac_th=0.1, ransac_max_iter=5000, PLOT=False):
    model_pts = []
    refine = []

    if PLOT:
        plt.figure()
        plt.imshow(dsm_crop, cmap='gray')
        plt.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')

    for idx, pp in enumerate(merged):
        if pp.geom_type == 'MultiPolygon':
            x = []
            y = []
            for geom in pp.geoms:
                x_, y_ = geom.boundary.coords.xy
                for i in range(len(x_)):
                    x.append(x_[i])
                    y.append(y_[i])
        else:
            x, y = pp.boundary.coords.xy

        if PLOT:
            plt.fill(x, y)  # , color=colors[idx,:])
            plt.plot(x, y, 'xk')
            plt.show(block=False)

        p3d_1 = p3d_all[pt_in_plane_all == merged_lb[idx], :]
        sup_lb = np.unique(pt_in_plane_supp[pt_in_plane_all == merged_lb[idx]])[0]

        if PLOT:
            show_planes(p3d_all, polypts_crop, pt_in_plane_supp == sup_lb, rows, cols)

        p3d_ = p3d_all[pt_in_plane_supp == sup_lb, :]
        this_model_pts = []
        plane = pyrsc.Plane()
        best_eq, best_inliers = plane.fit(p3d_1, ransac_th, maxIteration=ransac_max_iter)
        norm_to_z_angle = np.arccos(np.clip(np.dot([0, 0, 1], best_eq[0:3]), -1.0, 1.0)) * 180 / np.pi

        if 50 < norm_to_z_angle and norm_to_z_angle < 135:
            mean_z = ground_height  # too vertical plane are put to ground, or can be set as flat roof with mean height (i.e., np.mean(p3d_1[:,2]) )
            for i in range(len(x)):
                this_model_pts.append([x[i], y[i], mean_z])
        else:
            for i in range(len(x)):
                z = -(best_eq[0] * x[i] + best_eq[1] * y[i] + best_eq[3]) / best_eq[2]
                this_model_pts.append([x[i], y[i], z])
        if (p3d_1.shape[0] == p3d_.shape[0] and (p3d_1 == p3d_).all()):
            refine.append(-1)
        else:
            # if there are multiple plane in the same height-cluster, their plane equations should be refined to guarantee to have the same 3D junction points
            refine.append(sup_lb)
        model_pts.append(this_model_pts)

    # refine_u = np.unique(np.array(refine)).tolist()
    # refined_model_pts = []
    # for i in range(len(refine_u)):
    #     idx = np.where(np.array(refine)==refine_u[i])[0]    
    #     if(refine_u[i]!=-1):
    #         models = [np.array(model_pts[k]) for k in idx] 

    #         a = 0
    #     else:
    #         for k in idx:
    #             refined_model_pts.append(model_pts[k])

    return model_pts, refine


def get_triangular_meshes(model_pts, ground_height, polypts_crop, dsm_crop, PLOT=False):
    # - https://rufat.be/triangle/index.html (https://github.com/drufat/triangle)
    # - http://www.cs.cmu.edu/~quake/triangle.html
    # - https://stackoverflow.com/questions/66119717/how-do-i-triangulate-a-polygon-with-holes-without-extra-points
    # - https://stackoverflow.com/questions/67854771/efficient-triangulation-of-weakly-simple-polygon-with-known-ordered-boundary-po

    tri = []
    walls = []
    wall_tri = []

    for idx in range(len(model_pts)):
        if PLOT:
            plt.figure()
            plt.imshow(dsm_crop, cmap='gray')
            plt.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
        xyz = np.array(model_pts[idx])
        pts = []
        seg = []
        for i in range(xyz.shape[0] - 1):
            pts.append([xyz[i, 0], xyz[i, 1], xyz[i, 2]])
            if PLOT:
                plt.plot(xyz[i:i + 2, 0], xyz[i:i + 2, 1], 'x-')
                plt.show(block=False)
            if (i < xyz.shape[0] - 2):
                seg.append([i, i + 1])
            else:
                seg.append([i, 0])
        pts = np.array(pts)
        for s in seg:
            u1 = pts[s[0], :]
            u2 = pts[s[1], :]
            b1 = [u1[0], u1[1], ground_height]
            b2 = [u2[0], u2[1], ground_height]
            Wtri = np.array([[0, 1, 2], [2, 3, 0]])
            walls.append(np.vstack((u1, u2, b2, b1, u1)))
            wall_tri.append(Wtri)

        A = dict(vertices=pts[:, 0:2], segments=np.array(seg))
        B = tr.triangulate(A, opts='p')  # note that the origin uses 'qpa0.05' here     #  tringle_wrapper(A)
        tri.append(B['triangles'])

        if PLOT:
            tr.compare(plt, A, B)
            plt.show()

    return tri, walls, wall_tri


def save_output(save_path_glb, model_pts, tri, walls, wall_tri, X_zero, Y_zero, xmin, ymin, XLL, YLL, dsm, cellsize,
                crs=[3857]):
    trns = []
    trns_X_zero = []
    trns_Y_zero = []
    for t in range(len(crs)):
        tmp = Transformer.from_crs(3003, crs[t], always_xy=True)  # 3857     # 4326
        (X_zero_tmp, Y_zero_tmp) = tmp.transform(X_zero, Y_zero)
        trns.append(tmp)
        trns_X_zero.append(X_zero_tmp)
        trns_Y_zero.append(Y_zero_tmp)

    transformer_4326 = Transformer.from_crs(3003, 4326, always_xy=True)  # 3857     # 4326
    (lng_zero, lat_zero) = transformer_4326.transform(X_zero, Y_zero)

    roof_model_3003 = []
    wall_model_3003 = []
    roof_model_tri = []
    wall_model_tri = []

    trns_roof_model_pts = []
    trns_wall_model_pts = []
    for t in range(len(crs)):
        trns_roof_model_pts.append([])
        trns_wall_model_pts.append([])

    count = 0
    for idx in range(len(model_pts)):
        for x in model_pts[idx]:
            X, Y = coordinates_image_to_gis(x[0] + xmin, x[1] + ymin, XLL, YLL, np.shape(dsm)[0], cellsize / 2)
            roof_model_3003.append([X - X_zero, Y - Y_zero, x[2]])
            for t in range(len(crs)):
                (X_, Y_) = trns[t].transform(X, Y)
                trns_roof_model_pts[t].append([X_ - trns_X_zero[t], Y_ - trns_Y_zero[t], x[2]])
        for tidx in tri[idx]:
            t_ = tidx + count
            roof_model_tri.append([t_[2], t_[1], t_[0]])
        count = len(roof_model_3003)

    count = 0
    for idx in range(len(walls)):
        for x in walls[idx]:
            X, Y = coordinates_image_to_gis(x[0] + xmin, x[1] + ymin, XLL, YLL, np.shape(dsm)[0], cellsize / 2)
            wall_model_3003.append([X - X_zero, Y - Y_zero, x[2]])
            for t in range(len(crs)):
                (X_, Y_) = trns[t].transform(X, Y)
                trns_wall_model_pts[t].append([X_ - trns_X_zero[t], Y_ - trns_Y_zero[t], x[2]])
        for tidx in wall_tri[idx]:
            t_ = tidx + count
            wall_model_tri.append([t_[2], t_[1], t_[0]])
        count = len(wall_model_3003)

    # ratios = []
    # for i in range(len(full_model_3003)-1):
    #     pt0 = full_model_3003[i]
    #     pt1 = full_model_3003[i+1]
    #     pt0_ = full_model_3857[i]
    #     pt1_ = full_model_3857[i+1]
    #     d = sqrt(  (pt0[0] - pt1[0])*(pt0[0] - pt1[0]) + (pt0[1] - pt1[1])*(pt0[1] - pt1[1]) )
    #     d_ = sqrt(  (pt0_[0] - pt1_[0])*(pt0_[0] - pt1_[0]) + (pt0_[1] - pt1_[1])*(pt0_[1] - pt1_[1]) )
    #     if d != 0 and d_ != 0:
    #         ratios.append([d, d_, d/d_])
    # r = np.mean(np.array(ratios)[:,2])
    # full_model_3857_scaled = []
    # for i in range(len(full_model_3857)):
    #     pt = full_model_3857[i]        
    #     full_model_3857_scaled.append([pt[0], pt[1], pt[2]*r])

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_trisurf(np.array(full_model)[:,0], np.array(full_model)[:,1], np.array(full_model)[:,2], triangles=np.array(full_model_tri))
    # tmp = np.array(polypts)
    # ax.plot(tmp[:,0]-X_zero, tmp[:,1]-Y_zero, np.ones((tmp.shape[0]), dtype=float)*47, 'x-r')
    # plt.show(block=False)

    # export to glTF/GLB
    rgb_roof = [255.0, 0.0, 0.0, 1.0]
    rgb_wall = [255.0, 255.0, 204.0, 1.0]
    roof_vertices_3003 = np.array(roof_model_3003)
    roof_faces = np.array(roof_model_tri)
    save2glb(roof_vertices_3003, roof_faces, save_path_glb.split('.')[0] + '_3003_roof.glb', rgb_roof)
    wall_vertices_3003 = np.array(wall_model_3003)
    wall_faces = np.array(wall_model_tri)
    save2glb(wall_vertices_3003, wall_faces, save_path_glb.split('.')[0] + '_3003_wall.glb', rgb_wall)

    for t in range(len(crs)):
        roof_vertices_tmp = np.array(trns_roof_model_pts[t])
        save2glb(roof_vertices_tmp, roof_faces, save_path_glb.split('.')[0] + '_' + str(crs[t]) + '_roof.glb', rgb_roof)
        wall_vertices_tmp = np.array(trns_wall_model_pts[t])
        save2glb(wall_vertices_tmp, wall_faces, save_path_glb.split('.')[0] + '_' + str(crs[t]) + '_wall.glb', rgb_wall)

    tmp = save_path_glb.split('.')
    filename = tmp[0] + '.txt'
    with open(filename, 'w') as f:
        f.write('Origin point\n')
        f.write('crs ' + '3003' + '(' + str(X_zero) + ' , ' + str(Y_zero) + ')\n')
        f.write('crs ' + '4326' + ': lat = ' + str(lat_zero) + ' , lon = ' + str(lng_zero) + ')\n')
        for t in range(len(crs)):
            f.write('crs ' + str(crs[t]) + '(' + str(trns_X_zero[t]) + ' , ' + str(trns_Y_zero[t]) + ')\n')

    return None


#########################################################################################################################


read_geojson_plot = 0
dsm_crop_plot = 1
img_stat_plot = 0
region_growing_plot = 1
step_line_plot = 1
inner_plane_plot = 1
inner_line_plot = 1
plane_labeling_plot = 1
line_segment_plot = 1
filter_line_segment_plot = 0
polygon_patch_plot = 1
polygon_patch_label_plot = 1
pts3D_plot = 0
tri_mesh_plot = 0
model3D_plot = 1


def get_3D_model(data, dsm, dtm, XLL, YLL, cellsize, buildingID, X_zero, Y_zero, save_path_glb, times):

    start_main_time = time.time()

    # read shape polygon
    start_read_geojson_time = time.time()
    polypts_img, polypts, sub_polypts_img, sub_polypts = read_geojson_data(data, XLL, YLL, dsm, cellsize,
                                                                           PLOT=read_geojson_plot)
    end_read_geojson_time = time.time()

    # crop DSM on polygon
    margin = 5
    xmin = int(np.round(np.min(polypts_img[:, 0]) - margin))
    xmax = int(np.round(np.max(polypts_img[:, 0]) + margin))
    ymin = int(np.round(np.min(polypts_img[:, 1]) - margin))
    ymax = int(np.round(np.max(polypts_img[:, 1]) + margin))

    dsm_crop = dsm[ymin:ymax, xmin:xmax]
    dtm_crop = dtm[ymin:ymax, xmin:xmax]
    rows, cols = np.shape(dsm_crop)
    polypts_crop = polypts_img - [xmin, ymin]
    sub_polypts_crop = []
    for sp in range(len(sub_polypts_img)):
        sub_polypts_crop.append(sub_polypts_img[sp] - [xmin, ymin])

    if dsm_crop_plot:
        plt.figure()
        plt.imshow(dsm_crop, cmap='gray')
        plt.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
        plt.show(block=False)

    # remove some noise
    # dsm_crop = medfilt2d(dsm_crop, kernel_size=3)
    dsm_crop = cv2.bilateralFilter(dsm_crop, 3, 1, 1)

    # estimate gradient/edges
    start_get_stat_time = time.time()
    detector = ced.cannyEdgeDetector([dsm_crop], sigma=0.5, kernel_size=3, lowthreshold=0.09, highthreshold=0.10,
                                     weak_pixel=100)
    edge, grad, direc = detector.detect()

    # estimate normal vectors
    normals, angles = get_normal_vectors(dsm_crop, rows, cols)
    end_get_stat_time = time.time()

    PLOT_2 = img_stat_plot
    if PLOT_2:
        plt.figure()
        plt.imshow(edge[0], cmap='gray')
        plt.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
        plt.show(block=False)

        plt.figure()
        plt.imshow(grad[0], cmap='gray')
        plt.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
        plt.show(block=False)

        plt.figure()
        plt.imshow(direc[0], cmap='gray')
        plt.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
        plt.show(block=False)

        plt.figure()
        plt.imshow(normals)
        plt.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
        plt.show(block=False)

        plt.figure()
        plt.imshow(angles, cmap='gray')
        plt.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
        plt.show(block=False)

    # get array of 3D points (and stats) into the polygon
    p3d = []
    p3d_dtm = []
    nor = []
    grd = []
    drc = []
    poly = Polygon(polypts_crop)
    for x in range(np.shape(dsm_crop)[1]):
        for y in range(np.shape(dsm_crop)[0]):
            if (poly.contains(Point(x, y)) == True):
                p3d.append((x, y, dsm_crop[y, x]))
                p3d_dtm.append((x, y, dtm_crop[y, x]))
                nor.append((normals[y, x, 0], normals[y, x, 1], normals[y, x, 2]))
                grd.append(np.max(grad[0][y, x]))
                drc.append(direc[0][y, x])
    p3d = np.asarray(p3d)
    p3d_dtm = np.asarray(p3d_dtm)
    nor = np.asarray(nor)
    grd = np.asarray(grd)
    drc = np.asarray(drc)

    # cluster points on height using region growing
    start_rg_time = time.time()
    label, ulabel = region_growing_height_clustering(p3d, polypts_crop, sub_polypts_crop, rows, cols, th_h=0.25,
                                                     max_iter=100, PLOT=region_growing_plot)
    end_rg_time = time.time()

    # find step border
    start_sl_time = time.time()
    step_lines = get_step_lines(p3d, polypts_crop, label, ulabel, rows, cols, PLOT=step_line_plot)
    end_sl_time = time.time()

    # for each height cluster find inner planar sections and inner border lines ###############################################
    p3d_all = np.copy(p3d)
    nor_all = np.copy(nor)
    grd_all = np.copy(grd)
    drc_all = np.copy(drc)
    pt_in_plane_all = np.ones((p3d_all.shape[0]), dtype=int) * -1
    pt_in_plane_supp = np.ones((p3d_all.shape[0]), dtype=int) * -1

    inner_lines = []
    plane_cluster_time = []
    plane_line_time = []
    label_count = 0
    for k in range(len(ulabel)):

        p3d = p3d_all[label == ulabel[k], :]
        nor = nor_all[label == ulabel[k], :]
        grd = grd_all[label == ulabel[k]]
        drc = drc_all[label == ulabel[k]]

        if (p3d.shape[0] < 10):
            continue
        start_pl_time = time.time()
        pt_in_plane = get_inner_planes(p3d, polypts_crop, rows, cols, nor, grd, drc, PLOT=inner_plane_plot)
        end_pl_time = time.time()
        plane_cluster_time.append(end_pl_time - start_pl_time)

        pt_in_plane_all[label == ulabel[k]] = pt_in_plane + label_count
        pt_in_plane_supp[label == ulabel[k]] = ulabel[k]
        label_count += 100

        # compute internal border
        start_il_time = time.time()
        inner_lines = get_internal_lines(p3d, polypts_crop, pt_in_plane, inner_lines, rows, cols, PLOT=inner_line_plot)
        end_il_time = time.time()
        plane_line_time.append(end_il_time - start_il_time)
    ######################################################################################################
    PLOT_3 = plane_labeling_plot
    if PLOT_3:
        show_planes(p3d_all, polypts_crop, pt_in_plane_all, rows, cols)

    # merge all found lines and computes their intersections to find polygon patches
    lines = [*step_lines, *inner_lines]
    start_ls_time = time.time()
    edges, edges_label = extract_line_segments(lines, poly, dsm_crop, polypts_crop, sub_polypts_crop, PLOT=line_segment_plot)
    end_ls_time = time.time()

    start_fl_time = time.time()
    edges = filter_lines(edges, edges_label, poly, rows, cols, dsm_crop, polypts_crop, PLOT=filter_line_segment_plot)
    end_fl_time = time.time()

    start_pp_time = time.time()
    polygons = get_polygon_patches(edges, p3d_all, polypts_crop, label, rows, cols, PLOT=polygon_patch_plot)
    end_pp_time = time.time()

    #########################################################################################

    # polygon patch labeling

    start_ppl_time = time.time()
    # create a denser point set by super-sampling the dsm pixels. Each super-sample get the plane label of the original pixel
    p3d_superset, pt_in_plane_superset, pt_height_superset = get_label_superset(poly, p3d_all, pt_in_plane_all, label,
                                                                                delta=0.5, step=0.1)

    # assign a label to each polygon patch
    poly_label, u_poly_label = get_polygon_patch_label(polygons, p3d_superset, pt_height_superset, pt_in_plane_superset,
                                                       p3d_all, polypts_crop, pt_in_plane_all, rows, cols,
                                                       PLOT=polygon_patch_label_plot)

    # merge polygon patches with the same label into single plane patches
    merged, merged_lb = merge_polygon_patches(polygons, poly_label, u_poly_label)
    end_ppl_time = time.time()
    ##############################################################################################################à

    # 3D modeling
    start_3d_time = time.time()
    ground_height = np.mean(p3d_dtm[:, 2])

    # compute the Z-coordinate for the vertex of the plane patches
    model_pts, refine = get_3D_model_pts(p3d_all, pt_in_plane_all, pt_in_plane_supp, merged, merged_lb, ground_height,
                                         dsm_crop, polypts_crop, rows, cols, PLOT=pts3D_plot)

    # get triangular meshes for connecting the 3D points of each planar patch using Triangle:
    tri, walls, wall_tri = get_triangular_meshes(model_pts, ground_height, polypts_crop, dsm_crop, PLOT=tri_mesh_plot)
    end_3d_time = time.time()

    end_main_time = time.time()

    with open(times, 'a') as f:
        f.write('%s|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f\n' % (save_path_glb.split('.')[0],
                                                              end_main_time - start_main_time,
                                                              end_read_geojson_time - start_read_geojson_time,
                                                              end_get_stat_time - start_get_stat_time,
                                                              end_rg_time - start_rg_time,
                                                              end_sl_time - start_sl_time,
                                                              np.mean(np.array(plane_cluster_time)),
                                                              np.mean(np.array(plane_line_time)),
                                                              end_ls_time - start_ls_time,
                                                              end_fl_time - start_fl_time,
                                                              end_pp_time - start_pp_time,
                                                              end_ppl_time - start_ppl_time,
                                                              end_3d_time - start_3d_time))

    ########################################################################################################################
    # show the final results 
    PLOT_4 = model3D_plot
    if PLOT_4:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        for idx in range(len(model_pts)):
            xyz = np.array(model_pts[idx])
            ax.plot_trisurf(xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles=tri[idx])
        for idx in range(len(walls)):
            xyz = np.array(walls[idx])
            ax.plot_trisurf(xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles=wall_tri[idx])
        plt.show(block=False)

    # save model
    save_output(save_path_glb, model_pts, tri, walls, wall_tri, X_zero, Y_zero, xmin, ymin, XLL, YLL, dsm, cellsize)

    print("All done for " + buildingID + "!")


######################################################################################################################################

def get_3D_model_LOD1(data, dsm, dtm, XLL, YLL, cellsize, buildingID, X_zero, Y_zero,save_path_glb):

    # read shape polygon
    polypts_img, polypts, sub_polypts_img, sub_polypts = read_geojson_data(data, XLL, YLL, dsm, cellsize, PLOT=False)

    # crop DSM on polygon
    margin = 5
    xmin = int(np.round(np.min(polypts_img[:, 0]) - margin))
    xmax = int(np.round(np.max(polypts_img[:, 0]) + margin))
    ymin = int(np.round(np.min(polypts_img[:, 1]) - margin))
    ymax = int(np.round(np.max(polypts_img[:, 1]) + margin))

    dsm_crop = dsm[ymin:ymax, xmin:xmax]
    dtm_crop = dtm[ymin:ymax, xmin:xmax]
    rows, cols = np.shape(dsm_crop)
    polypts_crop = polypts_img - [xmin, ymin]
    sub_polypts_crop = []
    for sp in range(len(sub_polypts_img)):
        sub_polypts_crop.append(sub_polypts_img[sp] - [xmin, ymin])

    if dsm_crop_plot:
        plt.figure()
        plt.imshow(dsm_crop, cmap='gray')
        plt.plot(polypts_crop[:, 0], polypts_crop[:, 1], '-r')
        plt.show(block=False)

    # remove some noise
    # dsm_crop = medfilt2d(dsm_crop, kernel_size=3)
    dsm_crop = cv2.bilateralFilter(dsm_crop, 3, 1, 1)

    # get array of 3D points 
    p3d = []
    p3d_dtm = []
    poly = Polygon(polypts_crop)
    for x in range(np.shape(dsm_crop)[1]):
        for y in range(np.shape(dsm_crop)[0]):
            if (poly.contains(Point(x, y)) == True):
                p3d.append((x, y, dsm_crop[y, x]))
                p3d_dtm.append((x, y, dtm_crop[y, x]))

    p3d = np.asarray(p3d)
    p3d_dtm = np.asarray(p3d_dtm)
    roof_height = np.mean(p3d[:, 2])
    ground_height = np.mean(p3d_dtm[:, 2])

    # compute the Z-coordinate for the vertex of the plane patches
    model_pts = []
    for pt in polypts_crop:
        model_pts.append([pt[0], pt[1], roof_height])
    model_pts = [model_pts]
    refine = []

    # get triangular meshes for connecting the 3D points of each planar patch using Triangle:
    tri, walls, wall_tri = get_triangular_meshes(model_pts, ground_height, polypts_crop, dsm_crop, PLOT=tri_mesh_plot)

    # put everithing in a single model
    # save model
    save_output(save_path_glb, model_pts, tri, walls, wall_tri, X_zero, Y_zero, xmin, ymin, XLL, YLL, dsm, cellsize)

    print("All done for " + buildingID + "!")

###############################################################################################################################################
###############################################################################################################################################

# # Create the mesh
# cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(faces):
#     for j in range(3):
#         cube.vectors[i][j] = vertices[f[j],:]
# # Write the mesh to file "cube.stl"
# cube.save('TMP0_' + run_num + '.stl')

# # conversion

# o3d_mesh = o3d.io.read_triangle_mesh('TMP0_' + run_num + '.stl')
# # o3d_mesh_ = o3d.io.read_triangle_mesh(buildingID + '_.stl')
# o3d_mesh.compute_vertex_normals()
# o3d_mesh.paint_uniform_color([1, 0.706, 0])
# o3d.io.write_triangle_mesh('TMP1_' + run_num + '.stl', o3d_mesh)
# # o3d_mesh_.compute_vertex_normals()
# # o3d_mesh_.rotate(o3d_mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0)), center=(0, 0, 0))
# # o3d.visualization.draw_geometries([o3d_mesh])
# # o3d.io.write_triangle_mesh(buildingID + '.glb', o3d_mesh)
# # o3d.io.write_triangle_mesh(buildingID + '_rot.gltf', o3d_mesh_)

# stl2glb('TMP1_' + run_num + '.stl', save_path_glb)
