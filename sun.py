import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image
import open3d as o3d


class SUN:
    def __init__(self, meta_file='/home/lq/Documents/dataset/SUNRGBDMeta3DBB_v2.mat',rootPath='/home/lq/Documents/dataset'):
        self.rootPath=rootPath
        if not meta_file == None:
            print('loading metadata into memory...')
            tic = time.time()
            self.dataSet = sio.loadmat(meta_file)['SUNRGBDMeta'].ravel()
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
    
    def getSensorDataId(self,sensorType='kv1'):
        kv1Index=[]
        for i in range(len(self.dataSet)):
            if self.dataSet[i][8][0]==sensorType:
                kv1Index.append(i)
        return kv1Index


    def getPath(self,id):
        sequenceName=self.dataSet[id][0][0]
        imgPath=os.path.join(self.rootPath,sequenceName,self.dataSet[id][4][0].split('//')[1])
        depthPath=os.path.join(self.rootPath,sequenceName,self.dataSet[id][3][0].split('//')[1])
        segPath=os.path.join(self.rootPath,sequenceName,'seg.mat')
        return imgPath,depthPath,segPath

    def load3dPoints(self, id):
        """
        read points from certain room
        :param id: pos in metadata
        :return: 3d points
        """
        data = self.dataSet[id]
        sequenceName=data[0][0]
        depthPath=os.path.join(self.rootPath,sequenceName,data[3][0].split('//')[1])
        K=data[2]
        Rtilt=data[1]
        depthVis = Image.open(depthPath, 'r')
        depthVisData = np.asarray(depthVis, np.uint16)
        depthInpaint = np.bitwise_or(np.right_shift(depthVisData, 3), np.left_shift(depthVisData, 16 - 3))
        depthInpaint = depthInpaint.astype(np.single) / 1000
        depthInpaint[depthInpaint > 8] = 8
        points3d= self.load3dPoints_(depthInpaint, K)
        points3d = Rtilt.dot(points3d.T).T
        return points3d, depthInpaint

    def load3dPoints_(self, depth, K):
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]
        invalid = depth == 0
        x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        xw = (x - cx) * depth / fx
        yw = (y - cy) * depth / fy
        zw = depth
        points3dMatrix = np.stack((xw, zw, -yw), axis=2)
        points3dMatrix[np.stack((invalid, invalid, invalid), axis=2)] = np.nan
        points3d = points3dMatrix.reshape(-1, 3)
        return points3d

    def visPointCloud(self, id):
        points3d, depth = self.load3dPoints(id)
        # plt.title('depth')
        # plt.imshow(depth)
        # plt.show()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points3d)
        o3d.visualization.draw_geometries([pcd])
    


    def getImg(self,id):
        imgPath,depthPath,segPath=self.getPath(id)
        img=plt.imread(imgPath)
        depth=plt.imread(depthPath)
        seg= sio.loadmat(segPath)
        segLabel=seg['seglabel']
        segInstances=seg['seginstances']
        return img,depth,segLabel,segInstances

    def visImg(self, id):
        img,depth,segl,segi=self.getImg(id)
        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.subplot(2,2,2)
        plt.imshow(depth)
        plt.subplot(2,2,3)
        plt.imshow(segl)
        plt.subplot(2,2,4)
        plt.imshow(segi)

    def getCornerList(self,id):
        cornerList=[]
        data=self.dataSet[id][10].flatten()
        for i in range(len(data)):
            basis=data[i][0]
            coeffs=data[i][1][0]
            centroid=data[i][2]
            className=data[i][3]
            label=data[i][6]
            corner=self.getCorner(basis,coeffs,centroid)
            cornerList.append(corner)
        return cornerList

    def flip_toward_viewer(self,normals, points):
        points /= np.linalg.norm(points, axis=1)
        projection = np.sum(points * normals, axis=1)
        flip = projection > 0
        normals[flip] = - normals[flip]
        return normals

    def getCorner(self,basis,coeffs,centroid):
        corner = np.zeros((8, 3), dtype=np.float32)
        coeffs = coeffs.ravel()
        indices = np.argsort(- np.abs(basis[:, 0]))
        basis = basis[indices, :]
        coeffs = coeffs[indices]
        indices = np.argsort(- np.abs(basis[1:3, 1]))
        if indices[0] == 1:
            basis[[1, 2], :] = basis[[2, 1], :]
            coeffs[[1, 2]] = coeffs[[2, 1]]

        basis = self.flip_toward_viewer(basis, np.repeat(centroid, 3, axis=0))
        coeffs = abs(coeffs)
        corner[0] = -basis[0] * coeffs[0] + basis[1] * coeffs[1] + basis[2] * coeffs[2]
        corner[1] = basis[0] * coeffs[0] + basis[1] * coeffs[1] + basis[2] * coeffs[2]
        corner[2] = basis[0] * coeffs[0] + -basis[1] * coeffs[1] + basis[2] * coeffs[2]
        corner[3] = -basis[0] * coeffs[0] + -basis[1] * coeffs[1] + basis[2] * coeffs[2]

        corner[4] = -basis[0] * coeffs[0] + basis[1] * coeffs[1] + -basis[2] * coeffs[2]
        corner[5] = basis[0] * coeffs[0] + basis[1] * coeffs[1] + -basis[2] * coeffs[2]
        corner[6] = basis[0] * coeffs[0] + -basis[1] * coeffs[1] + -basis[2] * coeffs[2]
        corner[7] = -basis[0] * coeffs[0] + -basis[1] * coeffs[1] + -basis[2] * coeffs[2]
        corner += np.repeat(centroid, 8, axis=0)
        return corner





    def visCube(self,id,m=0):
        cornerList=self.getCornerList(id)
        lines = [[0, 1],[1, 2],[2, 3],[3, 0],[0, 4],[1, 5],[2, 6],
            [3, 7],[4, 5],[5, 6],[6, 7],[7, 4]]
        colors = [[0, 0, 1] for i in range(len(lines))]
        ll=[]
        for i in range(len(cornerList)):
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(cornerList[i]),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            ll.append(line_set)

        line_set.colors = o3d.utility.Vector3dVector(colors)

        if m==0:
            o3d.visualization.draw_geometries(ll)
        elif m==1:
            points3d, depth = self.load3dPoints(id)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points3d)
            ll.append(pcd)   
            o3d.visualization.draw_geometries(ll)  

class FrameData:
    def __init__(self, sequenceName='', groundTruth3DBB=None, Rtilt=None, K=None, depthpath='', rgbpath='',
                 anno_extrinsics=None, sensorType='', gtCorner3D=None):
        self.sequenceName = sequenceName
        self.groundTruth3DBB = groundTruth3DBB
        self.Rtilt = Rtilt
        self.K = K
        self.depthpath = depthpath
        self.rgbpath = rgbpath
        self.anno_extrinsics = anno_extrinsics
        self.sensorType = sensorType
        self.gtCorner3D = gtCorner3D

    def readFrame(self, framePath, dataRootPath='/data1/', cls=set(), bbmode='bb3d', bfx=False):
        if not os.path.isdir(framePath):
            pass  # TODO: throw exception

        self.sequenceName = self.getSequenceName(framePath, dataRootPath)
        self.sensorType = self.sequenceName.split(os.sep)[1]
        self.K = np.loadtxt(os.path.join(framePath, 'intrisics.txt')).reshape((3, 3))
        self.depthpath = os.path.join(framePath, 'depth_bfx') if bfx else os.path.join(framePath, 'depth')
        self.depthpath += os.listdir(self.depthpath)[0]
        self.rgbpath = os.path.join(framePath, 'image')
        self.rgbpath += os.listdir(self.rgbpath)[0]

        annotation_file = os.path.join(framePath, 'annotation3Dfinal', 'index.json')
        if os.path.isfile(annotation_file):
            with open(annotation_file, 'r') as f:
                annotateImage = json.load(f)
            for annotation in annotateImage['objects']:
                if annotation['name'] in {'wall', 'floor', 'ceiling'} or (len(cls) > 0 and annotation['name'] in cls):
                    continue
                box = annotation['polygon']

    def getSequenceName(self, thisPath, dataRoot='/data1/'):
        return os.path.relpath(thisPath, dataRoot)
