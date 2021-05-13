
# coding: utf-8

# 导入包
import time
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import random
from mnist import *


# 训练，验证，测试集和一些参数，beta1，beta2和epsilon都是Adam优化器里的参数
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
train, valid, test = read_data_sets("./dataset", one_hot=True)
x_train = train[0]
y_train = train[1]
x_valid = valid[0]
y_valid = valid[1]
x_test = test[0]
y_test = test[1]


# 调用gpu计算dsoftmax(不如numpy快)
@cuda.jit
def dsoftmax_gpu(da, z, dz):
    i, j = cuda.blockIdx.x, cuda.threadIdx.x
    cols = da.shape[1]
    tmp = 0
    for k in range(cols):
        if(j == k):
            tmp += da[i][j] * z[i][j] - da[i][k] * z[i][k] * z[i][j]
        else:
            tmp -= da[i][k] * z[i][k] * z[i][j]
    dz[i][j] = tmp


# 调用gpu计算全连接层(不如numpy快)
@cuda.jit
def linear_forward_gpu(x, w, b, z):
    i, j = cuda.blockIdx.x, cuda.threadIdx.x
    tmp = 0
    for k in range(w.shape[0]):
        tmp += x[i][k] * w[k][j]
    z[i][j] = tmp + b[0][j]


# 调用gpu计算全连接层反向(权重)(不如numpy快)
@cuda.jit
def linear_backward_gpu_dw(dw, a_preT, dz):
    i, j = cuda.blockIdx.x, cuda.threadIdx.x
    tmp = 0
    for k in range(a_preT.shape[1]):
        tmp += a_preT[i][k] * dz[k][j]
    dw[i][j] = tmp / dz.shape[0]


# 调用gpu计算全连接层反向(偏置)(不如numpy快)
@cuda.jit
def linear_backward_gpu_db(db, dz):
    j = cuda.threadIdx.x
    tmp = 0
    for k in range(dz.shape[0]):
        tmp += dz[k][j]
    db[0][j] = tmp / dz.shape[0]


# 调用gpu计算全连接层反向(da_pre)(不如numpy快)
@cuda.jit
def linear_backward_gpu_da_pre(da_pre, dz, wT):
    i, j = cuda.blockIdx.x, cuda.threadIdx.x
    tmp = 0
    for k in range(dz.shape[1]):
        tmp += dz[i][k] * wT[k][j]
    da_pre[i][j] = tmp


# 调用gpu计算卷积
@cuda.jit
def conv2d_forward_gpu(x_pad, k, b, stride, kh, kw, z):
    n, i = cuda.blockIdx.x, cuda.blockIdx.y
    j, l = cuda.threadIdx.x, cuda.threadIdx.y
    h_start = i * stride
    w_start = j * stride
    tmp = 0
    for ii in range(kh):
        for jj in range(kw):
            for kk in range(k.shape[3]):
                tmp += x_pad[n][h_start+ii][w_start+jj][kk] * k[l][ii][jj][kk]
    tmp += b[l][0]
    z[n][i][j][l] = tmp


# 调用gpu计算卷积反向(da_pre)
@cuda.jit
def conv2d_backward_gpu_da_pre(da_pre, k, dz, kh, kw):
    n, i = cuda.blockIdx.x, cuda.blockIdx.y
    j, l = cuda.threadIdx.x, cuda.threadIdx.y
    (m, h_pre, w_pre, c_pre) = da_pre.shape
    (m, h, w, c) = dz.shape
    h_start = (1 - kh) // 2
    h_end = (kh - 1) // 2 + 1
    w_start = (1 - kw) // 2
    w_end = (kw - 1) // 2 + 1
    kh2 = (kh - 1) // 2
    kw2 = (kw - 1) // 2
    tmp = 0
    # 行偏移
    for ii in range(h_start, h_end):
        # 列偏移
        for jj in range(w_start, w_end):
            h_index = ii + i
            w_index = jj + j
            if(h_index >= 0 and h_index < h and w_index >= 0 and w_index < w):
                for ll in range(k.shape[0]):
                    ki = kh2 - ii
                    kj = kw2 - jj
                    tmp += k[ll][ki][kj][l] * dz[n][h_index][w_index][ll]
    da_pre[n][i][j][l] = tmp


# 调用gpu计算卷积反向(卷积核)
@cuda.jit
def conv2d_backward_gpu_dk(dk, a_pre_pad, dz, kh, kw):
    c, kh = cuda.blockIdx.x, cuda.blockIdx.y
    kw, c_pre = cuda.threadIdx.x, cuda.threadIdx.y
    tmp = 0
    for i in range(dz.shape[1]):
        for j in range(dz.shape[2]):
            for n in range(a_pre_pad.shape[0]):
                tmp += a_pre_pad[n][i+kh][j+kw][c_pre] * dz[n][i][j][c]
    dk[c][kh][kw][c_pre] = tmp


# 调用gpu计算卷积反向(卷积层偏置)(不如numpy快)
@cuda.jit
def conv2d_backward_gpu_db(db, dz):
    c = cuda.blockIdx.x
    tmp = 0
    for i in range(dz.shape[0]):
        for j in range(dz.shape[1]):
            for k in range(dz.shape[2]):
                tmp += dz[i][j][k][c]
    db[c][0] = tmp


# 调用gpu计算池化
@cuda.jit
def pool2d_forward_gpu(x, size, stride, z):
    n, i = cuda.blockIdx.x, cuda.blockIdx.y
    j, l = cuda.threadIdx.x, cuda.threadIdx.y
    h_start = i * stride
    w_start = j * stride
    tmp = x[n][h_start][w_start][l]
    for ii in range(size):
        for jj in range(size):
            tmp = max(tmp, x[n][h_start+ii][w_start+jj][l])
    z[n][i][j][l] = tmp


# 调用gpu计算池化反向
@cuda.jit
def pool2d_backward_gpu(dz, a_pre, size, stride, da_pre):
    n, i = cuda.blockIdx.x, cuda.blockIdx.y
    j, l = cuda.threadIdx.x, cuda.threadIdx.y
    h_start = i * stride
    w_start = j * stride
    h_index = h_start
    w_index = w_start
    tmp = a_pre[n][h_start][w_start][l]
    for ii in range(size):
        for jj in range(size):
            if(a_pre[n][h_start+ii][w_start+jj][l] > tmp):
                h_index = h_start + ii
                w_index = w_start + jj
                tmp = a_pre[n][h_start+ii][w_start+jj][l]
    da_pre[n][h_index][w_index][l] = dz[n][i][j][l]


# 卷积神经网络
# x_size为输入的图片大小，conv2d_size为卷积核大小，pool2d_size为池化层大小，hidden_size为隐藏层的大小，y_size为分类个数
# conv2d_stride为卷积层步长，pool2d_stride为池化层步长，isgpu表示是否使用gpu运算(需要n卡和cuda)
# 例子:
# x_size = [28, 28, 1] ------ [h, w, d]
# conv2d_size = [[3, 3, 4], [3, 3, 8], [3, 3, 16]] ------ [h, w, n_C]
# h和w为卷积核的高度和宽度，n_C为卷积后图片的通道数
# conv2d_stride = [[1, 1], [1, 1], [1, 1]] ------ [h_stride, w_stride]
# 比如这里卷积后图片形状分别为[[28, 28, 4], [28, 28, 8], [28, 28, 16]]
# pool2d_size = [[2, 2], [2, 2], [2, 2]] ------ [h, w]
# pool2d_stride = [[2, 2], [2, 2], [2, 2]] ------ [h_stride, w_stride]
# hidden_size = [128, 64]
# y_size = 10
# 注意这里为了实现简单有以下参数必须遵守
# 默认卷积步长为1，卷积核高度和宽度一致，采用same mode，卷积后使用relu激活
# 默认池化不会出现交错，池化高度和宽度一致，池化步长高度和宽度一致
# 全连接层采用relu激活，最后一层使用softmax激活
# 默认使用Adam优化器，使用交叉熵
class CNN:
    def __init__(self, x_size, conv2d_size, conv2d_stride, pool2d_size, pool2d_stride, hidden_size, y_size, isgpu=False):
        self.isgpu = isgpu
        self.kernels = []				# 卷积核
        self.conv2d_bias = []			# 卷积层偏置
        self.kernels_stride = []		# 卷积步长
        self.pool2d_size = []			# 池化窗口大小
        self.pool2d_stride = []			# 池化步长
        self.weights = []				# 权重
        self.bias = []					# 偏置
        temp = conv2d_size[0]                           # 第一个卷积核形状
        # 由于规定了池化不会交错，这里可以根据输入图片的高度和宽度和每一次池化层的步长来计算最后图片的高度和宽度
        temph = x_size[0]				# 计算卷积后的高度
        tempw = x_size[1]				# 计算卷积后的宽度
        # 卷积层参数的初始化
        # 这里为了方便卷积核定义为4维张量，第一个维度为卷积后的图片通道数，第四个维度为卷积前的图片通道数
        # 第二，三个维度分别为高度和宽度，第一个卷积核需要x的通道数，因此不在循环中
        self.kernels.append(np.random.randn(temp[-1], temp[0], temp[1], x_size[-1]) * 0.01)
        self.conv2d_bias.append(np.ones([temp[-1], 1]) * 0.01)
        self.kernels_stride.append(conv2d_stride[0])
        self.pool2d_size.append(pool2d_size[0])
        self.pool2d_stride.append(pool2d_stride[0])
        temph = temph // pool2d_stride[0][0]
        tempw = tempw // pool2d_stride[0][1]
        for i in range(len(conv2d_size) - 1):
            # 每一次循环都是初始化第i+1个卷积核和偏置
            pre = conv2d_size[i]
            last = conv2d_size[i + 1]
            self.kernels.append(np.random.randn(last[-1], last[0], last[1], pre[-1]) * 0.01)
            self.conv2d_bias.append(np.ones([last[-1], 1]) * 0.01)
            self.kernels_stride.append(conv2d_stride[i + 1])
            self.pool2d_size.append(pool2d_size[i + 1])
            self.pool2d_stride.append(pool2d_stride[i + 1])
            temph = temph // pool2d_stride[i+1][0]
            tempw = tempw // pool2d_stride[i+1][1]
        # 全连接层参数的初始化，类似的需要前面那张的图片的大小，因此不在循环里
        self.weights.append(np.random.randn(temph * tempw * conv2d_size[-1][-1], hidden_size[0]) * 0.01)
        self.bias.append(np.ones([1, hidden_size[0]]) * 0.01)
        for i in range(len(hidden_size) - 1):
            # 每一次循环都是初始化第i+1个权重和偏置
            self.weights.append(np.random.randn(hidden_size[i], hidden_size[i + 1]) * 0.01)
            self.bias.append(np.ones([1, hidden_size[i + 1]]) * 0.01)
        self.weights.append(np.random.randn(hidden_size[-1], y_size) * 0.01)
        self.bias.append(np.ones([1, y_size]) * 0.01)
        
    def zero_pad(self, X, pad):
        # 对高度和宽度做零填充
        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        return X_pad

    def relu(self, x):
        # relu激活函数
        x[x < 0] = 0
        return x
    
    # da为激活后那一层的偏导，z为激活前那一层的数据
    def drelu(self, da, z):
        # relu激活函数所对应的导数
        dz = np.array(da)
        dz[z <= 0] = 0
        return dz
        
    def softmax(self, z):
        # softmax激活函数
        y = np.exp(z)
        fm = np.sum(y, axis=1, keepdims=True)
        fm = np.dot(fm, [[1] * z.shape[1]])
        return y / fm
    
    # da为激活后那一层的偏导，z为激活前那一层的数据
    def dsoftmax(self, da, z):
        # softmax激活函数所对应的导数
        dz = np.zeros(z.shape)
        # 比numpy慢没必要用gpu版本
        # if not self.isgpu:
        if True:
            for j in range(da.shape[1]):
                dz[:, j] += da[:, j] * z[:, j]
            for j in range(da.shape[1]):
                for k in range(da.shape[1]):
                    dz[:, j] -= da[:, k] * z[:, k] * z[:, j]
        else:
            da2 = cuda.to_device(da)
            z2 = cuda.to_device(z)
            dz2 = cuda.to_device(dz)
            dsoftmax_gpu[da.shape[0], da.shape[1]](da2, z2, dz2)
            cuda.synchronize()
            dz = dz2.copy_to_host()
        return dz
    
    # y_hat为预测概率，y为标签
    def loss(self, y_hat, y):
        # 交叉熵损失函数
        tmp = y * np.log(y_hat)
        L = -np.sum(tmp) / len(y)
        return L
    
    def dloss(self, y_hat, y):
        # 交叉熵损失函数所对应的导数
        return -y / y_hat / len(y)
    
    # x为输入，w为权重，b为偏置
    def linear_forward(self, x, w, b):
        # 全连接层前向传播
        z = np.zeros([x.shape[0], w.shape[1]])
        # 比numpy慢没必要用gpu版本
        # if not self.isgpu:
        if True:
            z = np.dot(x, w) + b
        else:
            x2 = cuda.to_device(x)
            w2 = cuda.to_device(w)
            b2 = cuda.to_device(b)
            z2 = cuda.to_device(z)
            linear_forward_gpu[x.shape[0], w.shape[1]](x2, w2, b2, z2)
            cuda.synchronize()
            z = z2.copy_to_host()
        return z
    
    # x为输入，w为权重，b为偏置，at为激活函数
    def linear_activation_forward(self, x, w, b, at):
        # 全连接层加激活函数前向传播
        z = self.linear_forward(x, w, b)
        if(at == "relu"):
            a = self.relu(z)
        elif(at == "softmax"):
            a = self.softmax(z)
        return z, a
    
    # dz为全连接层之后那一层的偏导，a_pre为全连接层之前的参数，w为权重，b为偏置
    def linear_backward(self, dz, a_pre, w, b):
        # 全连接层反向传播
        dw = np.zeros(w.shape)
        db = np.zeros(b.shape)
        da_pre = np.zeros(a_pre.shape)
        # 比numpy慢没必要用gpu版本
        # if not self.isgpu:
        if True:
            dw = np.dot(a_pre.T, dz) / len(a_pre)
            db = np.sum(dz, axis=0, keepdims=True) / len(a_pre)
            da_pre = np.dot(dz, w.T)
        else:
            dw2 = cuda.to_device(dw)
            db2 = cuda.to_device(db)
            da_pre2 = cuda.to_device(da_pre)
            a_preT2 = cuda.to_device(a_pre.T)
            dz2 = cuda.to_device(dz)
            wT2 = cuda.to_device(w.T)
            linear_backward_gpu_dw[dw.shape[0], dw.shape[1]](dw2, a_preT2, dz2)
            linear_backward_gpu_db[1, db.shape[1]](db2, dz2)
            linear_backward_gpu_da_pre[da_pre.shape[0], da_pre.shape[1]](da_pre2, dz2, wT2)
            cuda.synchronize()
            da_pre = da_pre2.copy_to_host()
            dw = dw2.copy_to_host()
            db = db2.copy_to_host()
        return da_pre, dw, db
    
    # da为激活后的那一层的偏导，z为全连接层之后的数据，a_pre为全连接层之前的数据，w为权重，b为偏置，at为激活函数
    def linear_activation_backward(self, da, w, b, z, a_pre, at):
        # 全连接层家激活函数反向传播
        if(at == "relu"):
            dz = self.drelu(da, z)
        elif(at == "softmax"):
            dz = self.dsoftmax(da, z)
        da_pre, dw, db = self.linear_backward(dz, a_pre, w, b)
        return da_pre, dw, db
    
    # a_slice为从原图片中剪切出来的一块，高度和宽度与卷积核一致，通道数为原图片的通道数，k为卷积核，b为偏置
    def conv2d_step(self, a_slice, k, b):
        # 单步卷积
        temp = a_slice * [k]
        s = temp.sum(axis=(1, 2, 3))
        z = s + [b]
        return z
    
    # x为输入图片，k为卷积核，b为偏置，stride为步长，pad为填充大小
    def conv2d_forward(self, x, k, b, stride, pad):
        # 卷积层前向传播
        (m, h_pre, w_pre, c_pre) = x.shape
        (c, kh, kw, c_pre) = k.shape
        h = int((h_pre - kh + 2 * pad) / stride) + 1		# 计算卷积后的高度
        w = int((w_pre - kw + 2 * pad) / stride) + 1		# 计算卷积后的宽度
        z = np.zeros([m, h, w, c])
        x_pad = self.zero_pad(x, pad)
        if not self.isgpu:
            for i in range(h):
                for j in range(w):
                    for l in range(c):
                        # 计算滑动窗口的左上角坐标和右下角坐标
                        h_start = i * stride
                        h_end = i * stride + kh
                        w_start = j * stride
                        w_end = j * stride + kw
                        a_slice = x_pad[:, h_start:h_end, w_start:w_end, :]		# 滑动窗口到每一个片段做卷积运算
                        z[:, i, j, l] = self.conv2d_step(a_slice, k[l], b[l])
        else:
            x_pad2 = cuda.to_device(x_pad)
            k2 = cuda.to_device(k)
            b2 = cuda.to_device(b)
            z2 = cuda.to_device(z)
            conv2d_forward_gpu[(m, h), (w, c)](x_pad2, k2, b2, stride, kh, kw, z2)
            cuda.synchronize()
            z = z2.copy_to_host()
        a = self.relu(z)
        return z, a
    
    # da为激活后那一层的偏导，a_pre为卷积层之前的数据，k为卷积核，b为偏置，z为卷积层之后的数据，stride为步长，pad为填充
    def conv2d_backward(self, da, a_pre, k, b, z, stride, pad):
        # 卷积层反向传播
        dz = self.drelu(da, z)
        (m, h_pre, w_pre, c_pre) = a_pre.shape
        (c, kh, kw, c_pre) = k.shape
        (m, h, w, c) = dz.shape
        da_pre = np.zeros([m, h_pre, w_pre, c_pre])
        dk = np.zeros([c, kh, kw, c_pre])
        db = np.zeros([c, 1])
        a_pre_pad = self.zero_pad(a_pre, pad)
        if not self.isgpu:
            da_pre_pad = self.zero_pad(da_pre, pad)
            # 直接按照原来的顺序循环，循环内部直接对单步卷积求偏导然后叠加即可，
            for i in range(h):
                for j in range(w):
                    for l in range(c):
                        # 计算滑动窗口的左上角坐标和右下角坐标
                        h_start = i * stride
                        h_end = i * stride + kh
                        w_start = j * stride
                        w_end = j * stride + kw
                        a_slice = a_pre_pad[:, h_start:h_end, w_start:w_end, :]		# 取出滑动窗口的每一个片段
                        da_pre_pad[:, h_start:h_end, w_start:w_end, :] += [k[l]] * dz[:, i:i+1, j:j+1, l:l+1]	# 与卷积核做点积
                        dk[l] += (a_slice * dz[:, i:i+1, j:j+1, l:l+1]).sum(axis = 0)		# 片段与上一个导数卷积
                        db[l] += dz[:, i, j, l].sum()										# 直接求和
            da_pre = da_pre_pad[:, pad:-pad, pad:-pad, :]
        else:
            k2 = cuda.to_device(k)
            dz2 = cuda.to_device(dz)
            dk2 = cuda.to_device(dk)
            # db部分numpy快一些
            # db2 = cuda.to_device(db)
            a_pre_pad2 = cuda.to_device(a_pre_pad)
            da_pre2 = cuda.to_device(da_pre)
            conv2d_backward_gpu_da_pre[(m, h_pre), (w_pre, c_pre)](da_pre2, k2, dz2, kh, kw)
            conv2d_backward_gpu_dk[(c, kh), (kw, c_pre)](dk2, a_pre_pad2, dz2, kh, kw)
            # conv2d_backward_gpu_db[c, 1](db2, dz2)
            cuda.synchronize()
            dk = dk2.copy_to_host()
            # db = db2.copy_to_host()
            db = dz.sum(axis=(0, 1, 2)).reshape(-1, 1)
            da_pre = da_pre2.copy_to_host()
        dk /= len(a_pre)
        db /= len(a_pre)
        return da_pre, dk, db
    
    # x为输入图片，size为池化大小，stride为池化步长
    def pool2d_forward(self, x, size, stride):
        # 池化层前向传播
        (m, h_pre, w_pre, c_pre) = x.shape
        h = int(1 + (h_pre - size) / stride)			# 计算汇聚后的高度
        w = int(1 + (w_pre - size) / stride)			# 计算汇聚后的宽度
        c = c_pre
        z = np.zeros([m, h, w, c])
        if not self.isgpu:
            for i in range(h):
                for j in range(w):
                    # 计算滑动窗口的左上角坐标和右下角坐标
                    h_start = i * stride
                    h_end = i * stride + size
                    w_start = j * stride
                    w_end = j * stride + size
                    x_slice = x[:, h_start:h_end, w_start:w_end, :]				# 滑动窗口到每一个片段计算(max pool)
                    z[:, i, j, :] = x_slice.max(axis=(1, 2))
        else:
            x2 = cuda.to_device(x)
            z2 = cuda.to_device(z)
            pool2d_forward_gpu[(m, h), (w, c)](x2, size, stride, z2)
            cuda.synchronize()
            z = z2.copy_to_host()
        return z
    
    # dz为池化后那一层偏导，a_pre为池化前那一层的数据，size为池化大小，stride为池化步长
    def pool2d_backward(self, dz, a_pre, size, stride):
        # 池化层反向传播
        (m, h_pre, w_pre, c_pre) = a_pre.shape
        (m, h, w, c) = dz.shape
        da_pre = np.zeros([m, h_pre, w_pre, c_pre])
        if not self.isgpu:
            for i in range(h):
                for j in range(w):
                    for l in range(c):
                        # 计算滑动窗口的左上角坐标和右下角坐标
                        h_start = i * stride
                        h_end = i * stride + size
                        w_start = j * stride
                        w_end = j * stride + size
                        a_pre_slice = a_pre[:, h_start:h_end, w_start:w_end, l]		# 取出滑动窗口的每一个片段
                        mask = (a_pre_slice == np.max(a_pre_slice))					# 记录最大值的位置
                        da_pre[:, h_start:h_end, w_start:w_end, l] += mask * dz[:, i:i+1, j:j+1, l]	# 只有最大值的位置有导数
        else:
            da_pre2 = cuda.to_device(da_pre)
            dz2 = cuda.to_device(dz)
            a_pre2 = cuda.to_device(a_pre)
            pool2d_backward_gpu[(m, h), (w, c)](dz2, a_pre2, size, stride, da_pre2)
            cuda.synchronize()
            da_pre = da_pre2.copy_to_host()
        return da_pre
        
    # 这里只实现默认情况下的前向传播，卷积步长为1，same mode，max pool，垂直和水平步长一致
    def predict_proba(self, x):
        # 整个模型的前向传播
        z_conv2d = []
        a_conv2d = []
        z_pool2d = []
        z_linear = []
        a_linear = []
        a = x
        # 卷积层前向传播
        for i in range(len(self.kernels)):
            k_temp = self.kernels[i]
            b_temp = self.conv2d_bias[i]
            stride = self.kernels_stride[i]
            # 这里为了方便实现默认步长为1
            pad = (k_temp.shape[1] - 1) // 2
            psize = self.pool2d_size[i]
            pstride = self.pool2d_stride[i]
            z0, z = self.conv2d_forward(a, k_temp, b_temp, stride[0], pad)
            a = self.pool2d_forward(z, psize[0], pstride[0])
            z_conv2d.append(z0)
            a_conv2d.append(z)
            z_pool2d.append(a)
        temp_shape = a.shape
        a = a.reshape(len(a), -1)
        # 全连接层前向传播
        for i in range(len(self.weights) - 1):
            z, a = self.linear_activation_forward(a, self.weights[i], self.bias[i], "relu")
            z_linear.append(z)
            a_linear.append(a)
        z, a = self.linear_activation_forward(a, self.weights[-1], self.bias[-1], "softmax")
        z_linear.append(z)
        a_linear.append(a)
        caches = [z_conv2d, a_conv2d, z_pool2d, z_linear, a_linear]
        return a, caches, temp_shape
    
    def predict(self, x):
        # 计算预测值
        y_hat, _, _ = self.predict_proba(x)
        return np.argmax(y_hat, axis=1)
        
    def grad(self, x, y):
        # 整个模型的反向传播
        dw = []		# 权重的导数
        db = []		# 偏置的导数
        dk = []		# 卷积核的导数
        dkb = []	# 卷积偏置的导数
        y_hat, caches, temp_shape = self.predict_proba(x)
        [z_conv2d, a_conv2d, z_pool2d, z_linear, a_linear] = caches
        da = self.dloss(y_hat, y)		# 计算loss对y_hat的偏导
        # 最后一层全连接层的各参数的导数
        da, dw0, db0 = self.linear_activation_backward(da, self.weights[-1], self.bias[-1]
                                                        , z_linear[-1], a_linear[-2], "softmax")
        dw.insert(0, dw0)
        db.insert(0, db0)
        # 第二层到倒数第二层的全连接层各参数的导数
        for i in range(2, len(self.weights)):
            da, dw0, db0 = self.linear_activation_backward(da, self.weights[-i], self.bias[-i]
                                                            , z_linear[-i], a_linear[-i-1], "relu")
            dw.insert(0, dw0)
            db.insert(0, db0)
        a_pre = z_pool2d[-1].reshape(len(x), -1)
        # 第一层全连接层各参数的导数
        da, dw0, db0 = self.linear_activation_backward(da, self.weights[0], self.bias[0], z_linear[0], a_pre, "relu")
        dw.insert(0, dw0)
        db.insert(0, db0)
        dz = da.reshape(temp_shape)
        # 第二层到最后一层卷积层和池化层的各参数的导数
        for i in range(1, len(self.kernels)):
            dz = self.pool2d_backward(dz, a_conv2d[-i], self.pool2d_size[-i][0], self.pool2d_stride[-i][0])
            stride = self.kernels_stride[-i]
            pad = (self.kernels[-i].shape[1] - 1) // 2
            dz, dk0, dkb0 = self.conv2d_backward(dz, z_pool2d[-i-1], self.kernels[-i], self.conv2d_bias[-i],
                                                 z_conv2d[-i], stride[0], pad)
            dk.insert(0, dk0)
            dkb.insert(0, dkb0)
        dz = self.pool2d_backward(dz, z_conv2d[0], self.pool2d_size[0][0], self.pool2d_stride[0][0])
        stride = self.kernels_stride[0]
        pad = (self.kernels[0].shape[1] - 1) // 2
        # 第一层卷积层和池化层的各参数的导数
        dz, dk0, dkb0 = self.conv2d_backward(dz, x, self.kernels[0], self.conv2d_bias[0],
                                            z_conv2d[0], stride[0], pad)
        dk.insert(0, dk0)
        dkb.insert(0, dkb0)
        return dw, db, dk, dkb
        
    def fit(self, x, y, x2, y2, lr=0.0005, epochs=1, batch_size=30, isgpu=False):
        # 拟合神经网络
        # lr为学习率，epochs为周期，batch_size为批次大小
        self.isgpu = isgpu
        history = [[], []]
        history2 = [[], []]
        wm = []		# 权重所对应的Adam优化器中需要迭代的第一个分量
        wv = []		# 权重所对应的Adam优化器中需要迭代的第二个分量
        bm = []		# 偏置所对应的Adam优化器中需要迭代的第一个分量
        bv = []		# 偏置所对应的Adam优化器中需要迭代的第二个分量
        km = []		# 卷积核所对应的Adam优化器中需要迭代的第一个分量
        kv = []		# 卷积核所对应的Adam优化器中需要迭代的第二个分量
        kbm = []	# 卷积层偏置所对应的Adam优化器中需要迭代的第一个分量
        kbv = []	# 卷积层偏置所对应的Adam优化器中需要迭代的第二个分量
        # 初始化为0
        for i in range(len(self.weights)):
            wv.append(np.zeros(self.weights[i].shape))
            wm.append(np.zeros(self.weights[i].shape))
            bv.append(np.zeros(self.bias[i].shape))
            bm.append(np.zeros(self.bias[i].shape))
        for i in range(len(self.kernels)):
            kv.append(np.zeros(self.kernels[i].shape))
            km.append(np.zeros(self.kernels[i].shape))
            kbv.append(np.zeros(self.conv2d_bias[i].shape))
            kbm.append(np.zeros(self.conv2d_bias[i].shape))
        for i in range(epochs):
            sample = random.sample(range(len(x)), len(x))		# 每个周期都将原来的样本打乱一次
            X = x[sample, :, :, :]
            Y = y[sample, :]
            last = time.time()
            for j in range(len(x) // batch_size):			# 分批次训练
                xx = X[batch_size * j : batch_size * (j + 1), :, :, :]
                yy = Y[batch_size * j : batch_size * (j + 1), :]
                if(j % 100 == 0):					# 每个周期记录一下loss和accuracy，以方便调参数
                    y_hat, _, _ = self.predict_proba(xx)
                    L = self.loss(y_hat, yy)								# 计算小批次训练集的损失
                    pred = np.argmax(y_hat, axis=1)
                    A = (pred == np.argmax(yy, axis=1)).sum() / len(yy)		# 计算小批次训练集的精确度
                    history[0].append(L)
                    history[1].append(A)
                    if(j % 200 == 0):
                        print('train: loss=%.3f' % L, end=" ")
                        print('accuracy=%.3f ' % A, end=" ")
                    y_hat, _, _ = self.predict_proba(x2)
                    L = self.loss(y_hat, y2)								# 计算验证集的损失
                    pred = np.argmax(y_hat, axis=1)
                    A = (pred == np.argmax(y2, axis=1)).sum() / len(y2)		# 计算验证集的精确度
                    history2[0].append(L)
                    history2[1].append(A)
                    if(j % 200 == 0):
                        print('valid: loss=%.3f' % L, end=" ")
                        print('accuracy=%.3f ' % A, end=" ")
                        print('time=%.3fs' % (time.time() - last))
                        last = time.time()
                # s = time.time()
                dw, db, dk, dkb = self.grad(xx, yy)				# 计算梯度
                # print(time.time() - s)
                for k in range(len(dw)):				# 根据梯度更新Adam优化器中的两个分量
                    wm[k] = beta1 * wm[k] + (1 - beta1) * dw[k]
                    wv[k] = beta2 * wv[k] + (1 - beta2) * dw[k] ** 2
                    bm[k] = beta1 * bm[k] + (1 - beta1) * db[k]
                    bv[k] = beta2 * bv[k] + (1 - beta2) * db[k] ** 2
                for k in range(len(dw)):				# 计算该次迭代Adam优化器中最终使用的两个分量
                    wm_hat = wm[k] / (1 - beta1 ** (i + 1))
                    wv_hat = wv[k] / (1 - beta2 ** (i + 1))
                    bm_hat = bm[k] / (1 - beta1 ** (i + 1))
                    bv_hat = bv[k] / (1 - beta2 ** (i + 1))
                    self.weights[k] -= lr * wm_hat / (np.sqrt(wv_hat) + epsilon)	# 对权重迭代
                    self.bias[k] -= lr * bm_hat / (np.sqrt(bv_hat) + epsilon)		# 对偏置迭代
                for k in range(len(dk)):				# 根据梯度更新Adam优化器中的两个分量
                    km[k] = beta1 * km[k] + (1 - beta1) * dk[k]
                    kv[k] = beta2 * kv[k] + (1 - beta2) * dk[k] ** 2
                    kbm[k] = beta1 * kbm[k] + (1 - beta1) * dkb[k]
                    kbv[k] = beta2 * kbv[k] + (1 - beta2) * dkb[k] ** 2
                for k in range(len(dk)):				# 计算该次迭代Adam优化器中最终使用的两个分量
                    km_hat = km[k] / (1 - beta1 ** (i + 1))
                    kv_hat = kv[k] / (1 - beta2 ** (i + 1))
                    kbm_hat = kbm[k] / (1 - beta1 ** (i + 1))
                    kbv_hat = kbv[k] / (1 - beta2 ** (i + 1))
                    self.kernels[k] -= lr * km_hat / (np.sqrt(kv_hat) + epsilon)		# 对卷积核迭代
                    self.conv2d_bias[k] -= lr * kbm_hat / (np.sqrt(kbv_hat) + epsilon)	# 对卷积层偏置迭代
        return history, history2


# 调用gpu运行指令:
# CUDA_VISIBLE_DEVICES='0' python mnist_cnn_gpu.py
if __name__ == "__main__":
    # 训练
    # 无n卡或cuda的训练时将最后一个True改为False，建议将卷积核调小一些，比如[[3, 3, 1]]，隐藏层调小一些，比如[32]，否则运行会非常慢
    x_size = [28, 28, 1]
    conv2d_size = [[5, 5, 16]]
    conv2d_stride = [[1, 1]]
    pool2d_size = [[2, 2]]
    pool2d_stride = [[2, 2]]
    hidden_size = [128]
    y_size = 10
    model = CNN(x_size, conv2d_size, conv2d_stride, pool2d_size, pool2d_stride, hidden_size, y_size)
    H, H2 = model.fit(x_train, y_train, x_valid, y_valid, 0.0005, 1, 30, True)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(H[0])
    plt.plot(H2[0])
    plt.subplot(1, 2, 2)
    plt.plot(H[1])
    plt.plot(H2[1])
    plt.show()

    # 测试
    pred = model.predict(x_test)
    acc = (pred == np.argmax(y_test, axis=1)).sum() / len(pred)
    print("test accuracy=: ", acc)
