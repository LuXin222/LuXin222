# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ..losses import ProbIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_, vector_
from ppdet.modeling.backbones.cspresnet import ConvBNLayer
from ppdet.modeling.ops import get_static_shape, get_act_fn, anchor_generator
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['PPYOLOERHead']# 列表定义了只有 PPYOLOERHead 这个类应该在使用 from 模块名 import * 语法导入时被导入，而其他成员不会被导入。这是一种显式控制导入的方式，
# 通常用于防止不必要的全局变量和函数污染导入命名空间。如果模块中有其他类、函数、变量等需要被导入，它们必须显式地通过 from 模块名 import 具体成员名 的方式导入。这有助于代码的可维护性和可读性。


class ESEAttn(nn.Layer):
    def __init__(self, feat_channels, act='swish'):#这是 ESEAttn 类的构造函数，接收特征通道数 feat_channels 和激活函数类型 act（默认为 'swish'）作为参数。
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2D(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.01)#权重初始化

    def forward(self, feat, avg_feat):#feat表示输入的特征数据，而 avg_feat 可能是一些平均特征或其他辅助信息。
        weight = F.sigmoid(self.fc(avg_feat))#这是一个全连接层的前向传播操作，它将输入 avg_feat 通过全连接层进行线性变换，产生一个输出，这是对前面全连接层输出的应用 sigmoid 激活函数它可以将输出限制在 0 和 1 之间
        return self.conv(feat * weight)#这是将输入特征 feat 与前面计算得到的权重 weight 逐元素相乘。这样可以对输入特征进行加权，以便在后续的卷积操作中进行调整。


@register
class PPYOLOERHead(nn.Layer):
    __shared__ = ['num_classes', 'trt']#这表示这两个成员是在不同的类之间共享的，它们的值在不同的类实例之间是相同的。这种共享可以使多个子类共享相同的属性值，以减少代码的冗余和提高可维护性。
    __inject__ = ['static_assigner', 'assigner', 'nms']
    #属性定义了哪些类成员需要通过依赖注入的方式提供给类。依赖注入是一种设计模式，它允许将依赖关系传递给一个对象，而不是在对象内部硬编码这些依赖关系。在这个示例中，__inject__ 列表中包含了
    # 'static_assigner'、'assigner' 和 'nms' 三个成员名称。这表示这三个成员需要从外部注入，它们的具体实现通常由类的用户提供。这种机制使类的行为更加可配置和灵活，
    # 允许用户自定义这些关键组件的实现。
    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=15,
                 act='swish',
                 fpn_strides=(32, 16, 8),##通常用于表示特征金字塔网络（Feature Pyramid Network，FPN）中不同层级特征图的步幅。步幅是指在图像上滑动卷积窗口时每次移动的像素数。
                 grid_cell_offset=0.5,#假设有一个图像被划分成一个10x10的网格，其中每个网格单元的大小是10x10像素。如果 grid_cell_offset 的值为0.5，
                 # 那么表示每个网格单元的中心都位于该区域的中心位置，即在每个网格单元的左上角和右下角都各有一个距离边界5像素的偏移。
                 angle_max=90,
                 use_varifocal_loss=True,#变焦损失
                 static_assigner_epoch=4,#静态分配器
                 trt=False,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 loss_weight={'class': 1.0,
                              'iou': 2.5,#表示IoU损失（Intersection over Union loss）的权重为2.5。IoU损失通常用于衡量模型对目标边界框（bounding box）的精确性，即目标边界框与真实边界框的重叠度
                              #假设你正在训练一个目标检测模型，该模型的任务是检测图像中的汽车。在训练期间，
                              # 你使用了多个损失函数来优化模型，其中之一是IoU损失。如果你将IoU损失的权重设置为2.5，
                              # 而将其他损失的权重设置为1.0（或其他值），那么模型在训练过程中将更加关注IoU损失。
                              'dfl': 0.05}):
        super(PPYOLOERHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_offset = grid_cell_offset
        self.angle_max = angle_max
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.half_pi = paddle.to_tensor(
            [1.5707963267948966], dtype=paddle.float32)
        self.half_pi_bin = self.half_pi / angle_max
        #所以，整行代码的目的是将半圆的弧度值（π/2）分割成多个小单位，每个单位的大小由 angle_max 决定，然
        # 后将结果赋给 self.half_pi_bin 变量以供后续使用。这通常在处理角度相关的任务或计算中用于将连续的角度范围划分为离散的部分。
        self.iou_loss = ProbIoULoss()
        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        # stem
        self.stem_cls = nn.LayerList()
        self.stem_reg = nn.LayerList()
        self.stem_angle = nn.LayerList()
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        self.trt = trt
        for in_c in self.in_channels:#: 遍历 self.in_channels 列表。这个列表可能包含输入通道数。
            self.stem_cls.append(ESEAttn(in_c, act=act))#对于每个输入通道数 in_c，创建一个 ESEAttn 层（之前定义过的注意力层），并将其添加到 self.stem_cls 列表中。
            self.stem_reg.append(ESEAttn(in_c, act=act))
            self.stem_angle.append(ESEAttn(in_c, act=act))
        # pred head
        self.pred_cls = nn.LayerList()#初始化 self.pred_cls 为一个空的层列表，用于分类（classification）预测。
        self.pred_reg = nn.LayerList()
        self.pred_angle = nn.LayerList()
        for in_c in self.in_channels:#: 遍历 self.in_channels 列表。这个列表可能包含输入通道数。
            self.pred_cls.append(
                nn.Conv2D(
                    in_c, self.num_classes, 3, padding=1))#对于每个输入通道数 in_c，创建一个卷积层用于分类预测，并添加到 self.pred_cls。
            self.pred_reg.append(nn.Conv2D(in_c, 4, 3, padding=1))
            self.pred_angle.append(
                nn.Conv2D(
                    in_c, self.angle_max + 1, 3, padding=1))
        self.angle_proj_conv = nn.Conv2D(
            self.angle_max + 1, 1, 1, bias_attr=False)#创建二维卷积层
        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):# 这是一个类方法（由装饰器 @classmethod 隐含，尽管在代码中没有显式显示）。cls 表示类本身，cfg 是传入的配置对象，input_shape 是输入形状信息。
        return {'in_channels': [i.channels for i in input_shape], }
#方法返回一个字典，这个字典包含了创建类实例所需的参数。特别是，它包含了一个键 in_channels：
    #rom_config 方法根据提供的配置和输入形状信息生成了一个字典，这个字典可以用来初始化类的实例。
    # 这种方法通常在基于配置文件构建复杂模型时非常有用，因为它允许灵活地根据配置创建不同的网络结构或组件。
    # 在实际使用中，这个方法可能会被进一步用来实例化特定的网络层或整个模型的一部分。
    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)#在目标检测任务中，大多数区域通常不包含目标，因此设置一个小的先验概率可以使模型在初始阶段倾向于
        # 预测“无目标”的类别。函数用于根据给定的先验概率（prior_prob）初始化卷积层或全连接层的偏置值。
        bias_angle = [10.] + [1.] * self.angle_max
        for cls_, reg_, angle_ in zip(self.pred_cls, self.pred_reg,#遍历三个网络层列表（分类、回归和角度预测）
                                      self.pred_angle):
            normal_(cls_.weight, std=0.01)# 使用标准差为0.01的正态分布初始化分类层的权重。
            constant_(cls_.bias, bias_cls)#将分类层的偏置初始化为 bias_cls 的值。
            normal_(reg_.weight, std=0.01)
            constant_(reg_.bias)
            constant_(angle_.weight)#初始化角度预测层的权重（具体值未指定）
            vector_(angle_.bias, bias_angle)# 将角度预测层的偏置初始化为 bias_angle 列表的值。

        angle_proj = paddle.linspace(0, self.angle_max, self.angle_max + 1)#PaddlePaddle 深度学习框架这个表达式调用了 PaddlePaddle 的 linspace
        # 函数来生成一个等间距数列。您的代码上下文中，这个数列可能用于表示不同的角度类别或者用于后续的角度计算。例如，如果这是一个目标检测模型，这个数列可能代表用于角度预测的不同类别。
        self.angle_proj = angle_proj * self.half_pi_bin
        self.angle_proj_conv.weight.set_value(
            self.angle_proj.reshape([1, self.angle_max + 1, 1, 1]))
        self.angle_proj_conv.weight.stop_gradient = True

    def _generate_anchors(self, feats):#这是一个类方法，用于根据输入的特征图（feats）生成锚点。
        if self.trt:
            anchor_points = []# self.trt 是否为真。如果是，这通常意味着模型正在使用 TensorRT 进行优化。
            for feat, stride in zip(feats, self.fpn_strides):#遍历输入的特征图（feats）和特征金字塔网络（FPN）的步幅（self.fpn_strides）。
                # 示例zip函数
                # list1 = [1, 2, 3]
                # list2 = ['a', 'b', 'c']
                # paired = zip(list1, list2)
                # for item in paired:
                #     print(item)
                # (1, 'a')
                # (2, 'b')
                # (3, 'c')
                _, _, h, w = paddle.shape(feat)
                anchor, _ = anchor_generator(
                    feat,
                    stride * 4,
                    1.0, [1.0, 1.0, 1.0, 1.0], [stride, stride],
                    offset=0.5)#。这个函数可能使用特征图、步幅和其他参数来计算锚点的位置
                x1, y1, x2, y2 = paddle.split(anchor, 4, axis=-1)#将锚点分割为左上角（x1, y1）和右下角（x2, y2）的坐标。
                xc = (x1 + x2 + 1) / 2#就锚点中心坐标
                yc = (y1 + y2 + 1) / 2
                anchor_point = paddle.concat(
                    [xc, yc], axis=-1).reshape((1, h * w, 2))
                anchor_points.append(anchor_point)
            anchor_points = paddle.concat(anchor_points, axis=1)#将中心坐标 xc 和 yc 拼接起来，并调整形状以匹配期望的输出格式。
            # 示例
            # 假设 xc 和 yc 是计算出的中心坐标
            # 例如，有三个锚点的 x 和 y 坐标
            # xc = paddle.to_tensor([2.5, 3.5, 4.5])
            # yc = paddle.to_tensor([1.5, 2.5, 3.5])
            # 使用 paddle.concat 将 xc 和 yc 拼接起来
            # axis=-1 表示在最后一个维度上进行拼接，即将每个点的 x 和 y 坐标配对
            # anchor_point = paddle.concat([xc, yc], axis=-1)
            # print(anchor_point)
            # [[2.5, 1.5],
            #  [3.5, 2.5],
            #  [4.5, 3.5]]
            return anchor_points, None, None
        else:
            anchor_points = []
            stride_tensor = []
            num_anchors_list = []
            for feat, stride in zip(feats, self.fpn_strides):
                _, _, h, w = paddle.shape(feat)
                shift_x = (paddle.arange(end=w) + 0.5) * stride
                shift_y = (paddle.arange(end=h) + 0.5) * stride
                shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
                anchor_point = paddle.cast(
                    paddle.stack(
                        [shift_x, shift_y], axis=-1), dtype='float32')
                anchor_points.append(anchor_point.reshape([1, -1, 2]))
                stride_tensor.append(
                    paddle.full(
                        [1, h * w, 1], stride, dtype='float32'))
                num_anchors_list.append(h * w)
            anchor_points = paddle.concat(anchor_points, axis=1)
            stride_tensor = paddle.concat(stride_tensor, axis=1)
            return anchor_points, stride_tensor, num_anchors_list

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:#todo 走这
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def forward_train(self, feats, targets):
        anchor_points, stride_tensor, num_anchors_list = self._generate_anchors(
            feats)#todo return anchor_points, None, None该方法用于生成锚框（anchor boxes）的位置、步幅（stride）信息和锚框的数量。这些信息在目标检测中用于计算损失函数和生成预测框。

        cls_score_list, reg_dist_list, reg_angle_list = [], [], []#创建三个空列表，分别用于存储分类得分（cls_score）、距离回归（reg_dist）、角度回归（reg_angle）的结果。
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))#对当前特征图进行自适应平均池化，将特征图的尺寸降为 1x1，以提取全局特征。
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)#通过卷积神经网络模型（self.stem_cls[i]）ESEAttncengc层每个入通道的特征图进行处理，然后将其与原始特征图相加，最后经过分类头部模型（self.pred_cls[i]）nn.Conv2D得到分类得分 cls_logit。
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))#同样地，通过卷积神经网络模型（self.stem_reg[i]）对特征图进行处理，然后经过距离回归头部模型（self.pred_reg[i]）得到距离回归结果 reg_dist
            reg_angle = self.pred_angle[i](self.stem_angle[i](feat, avg_feat))#通过卷积神经网络模型（self.stem_angle[i]）对特征图进行处理，然后经过角度回归头部模型（self.pred_angle[i]）得到角度回归结果
            # cls and reg
            cls_score = F.sigmoid(cls_logit)#sigmod主要是非线性建模能力#将分类得分 cls_logit 经过 sigmoid 函数处理，将其转换为概率得分 cls_score。
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))#将各个头部的分类得分、距离回归结果和角度回归结果分别展平（flatten）并转置（transpose）后，添加到对应的列表中。这样处理后的结果将用于计算损失函数。
            reg_dist_list.append(reg_dist.flatten(2).transpose([0, 2, 1]))
            reg_angle_list.append(reg_angle.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)#将所有头部的分类得分、距离回归结果和角度回归结果连接在一起，得到一个完整的预测结果。
        reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        reg_angle_list = paddle.concat(reg_angle_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_dist_list, reg_angle_list, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)
#########################################模型的推断（evaluation）过程，用于在测试时使用模型进行前向推断，获取分类和回归的预测结果
    def forward_eval(self, feats):
        cls_score_list, reg_box_list = [], []
        anchor_points, _, _ = self._generate_anchors(feats)
        for i, (feat, stride) in enumerate(zip(feats, self.fpn_strides)):
            b, _, h, w = paddle.shape(feat)#取当前特征图的形状信息，其中 b 表示批量大小，h 表示高度，w 表示宽度。
            l = h * w#计算当前特征图的像素点数量，即特征图的高度乘以宽度。
            # cls
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))#对特征图进行自适应平均池化，将特征图的大小调整为 (1, 1)。
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)#通过模型的分类头部进行分类预测，stem_cls[i] 和 pred_cls[i] 是模型中的特定子模块，用于处理特征图并生成分类得分
            # reg
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_xy, reg_wh = paddle.split(reg_dist, 2, axis=1)#将回归预测 reg_dist 沿着通道维度分割成两部分，分别表示坐标偏移 reg_xy 和宽度-高度偏移 reg_wh。
            reg_xy = reg_xy * stride
            reg_wh = (F.elu(reg_wh) + 1.) * stride
            reg_angle = self.pred_angle[i](self.stem_angle[i](feat, avg_feat))
            reg_angle = self.angle_proj_conv(F.softmax(reg_angle, axis=1))
            reg_box = paddle.concat([reg_xy, reg_wh, reg_angle], axis=1)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)#将分类得分的 logits cls_logit 经过 Sigmoid 函数处理，将其转换为概率得分 cls_score。Sigmoid 函数可以将输入值映射到0到1之间的概率值。
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_box_list.append(reg_box.reshape([b, 5, l]))#将当前特征层上的回归边界框信息张量添加到 reg_box_list 中，并对其进行形状变换。这里将形状变换为 [batch_size, 5, l]，
            # 其中 5 表示每个边界框包含 5 个属性（x、y、宽度、高度、角度），l 表示特征图上的像素数量。

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_box_list = paddle.concat(reg_box_list, axis=-1).transpose([0, 2, 1])
        reg_xy, reg_wha = paddle.split(reg_box_list, [2, 3], axis=-1)
        reg_xy = reg_xy + anchor_points
        reg_box_list = paddle.concat([reg_xy, reg_wha], axis=-1)#沿着最后一个维度切分成两部分。reg_xy 包含前两个属性，即边界框的 x 和 y 坐标信息，reg_wha 包含后三个属性，即边界框的宽度、高度和角度信息。
        return cls_score_list, reg_box_list#分别表示分类得分和回归边界框信息。这些信息将用于后续的目标检测结果的生成和后处理。

    def _bbox_decode(self, points, pred_dist, pred_angle, stride_tensor):#是一个张量，表示特征图上的每个点的中心坐标。
        #pred_dist是模型预测的距离信息，通常包括目标的中心点坐标和宽高信息。这个张量的形状是 [batch_size, anchor_num, height, width, 4]，其中 4 表示包括 x、y、宽度和高度四个预测值。
        #pred_angle是模型预测的角度信息，通常包括目标的旋转角度信息。这个张量的形状是 [batch_size, anchor_num, height, width, angle_max + 1]，其中 angle_max 表示角度的最大类别数，加 1 是因为通常还有一个类别表示不同角度的背景。
        # stride_tensor是一个张量，表示特征图的步幅信息，用于将预测结果从特征图尺度映射回原始图像尺度。
        # predict vector（向量） to x, y, w, h, angle
        b, l = pred_angle.shape[:2]#获取 pred_angle 张量的形状，其中 b 表示 batch size，l 表示 anchor 数量。
        xy, wh = paddle.split(pred_dist, 2, axis=-1)#拆分 pred_dist 张量，将前两个通道（通常表示 x 和 y 偏移）拆分为 xy，将后两个通道（通常表示宽度和高度）拆分为 wh。
        xy = xy * stride_tensor + points#将 xy 张量乘以步幅 stride_tensor 并加上特征图上的点坐标 points，从而得到目标的中心坐标。
        wh = (F.elu(wh) + 1.) * stride_tensor#对 wh 张量进行非线性激活函数 ELU（Exponential Linear Unit）操作，然后加上 1 并乘以步幅 stride_tensor，从而得到目标的宽度和高度。
        angle = F.softmax(pred_angle.reshape([b, l, 1, self.angle_max + 1#对 pred_angle 张量进行 Softmax 操作，将其转换为角度概率分布。这里的 reshape 操作将 pred_angle 张量的形状变为 [batch_size, anchor_num, 1, angle_max + 1]。
                                              ])).matmul(self.angle_proj)#将上一步得到的角度概率分布与 self.angle_proj 矩阵相乘，这个矩阵通常用于将角度概率转换为实际的角度值。
        return paddle.concat([xy, wh, angle], axis=-1)

    def get_loss(self, head_outs, gt_meta):#todo gt_meta是targets
        pred_scores, pred_dist, pred_angle, \
        anchor_points, num_anchors_list, stride_tensor = head_outs
        # [B, N, 5] -> [B, N, 5]
        pred_bboxes = self._bbox_decode(anchor_points, pred_dist, pred_angle,
                                        stride_tensor)#forward_train的输出
        gt_labels = gt_meta['gt_class']#gt_labels = gt_meta['gt_class']：从名为 gt_meta 的字典中获取真实标签的类别信息。这个操作假设 gt_meta 字典包含了真实标签的相关信息，其中 'gt_class' 键对应的值是一个形状为 [B, N] 的张量，其中 B 表示 batch size，
        # N 表示每个样本（图像）中的目标数量。这个张量存储了每个目标的类别标签，其中 [i, j] 元素表示第 i 个样本中第 j 个目标的类别。
        # [B, N, 5]
        gt_bboxes = gt_meta['gt_rbox']
        #从 gt_meta 字典中获取真实标签的边界框信息。同样，这个操作假设 'gt_rbox' 键对应的值是一个形状为 [B, N, 5] 的张量，其中 B 表示 batch size，N 表示每个样本中的目标数量，
        # 5 表示每个目标的边界框信息。通常，这个张量的最后一维包含了目标的中心坐标、宽度、高度和旋转角度等信息，这些信息用于描述目标在图像中的位置和形状。
        pad_gt_mask = gt_meta['pad_gt_mask']
        #这个部分代码可能是为了处理在某些情况下，每个图像中的目标数量不一致的情况。'pad_gt_mask' 键对应的值是一个形状为 [B, N] 的二进制掩码（mask）张量，其中 B 表示 batch size，
        # N 表示每个样本中的目标数量。这个掩码用于标识哪些位置是有效的目标位置，哪些位置是填充（pad）的无效位置。这对于批量处理具有不同目标数量的图像非常有用。
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:#4这段代码主要根据条件选择使用不同的目标分配策略，从而确定哪些预测的锚框（anchors）与真实目标匹配。
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchor_points,
                    stride_tensor,
                    num_anchors_list,
                    gt_labels,
                    gt_meta['gt_bbox'],
                    gt_bboxes,
                    pad_gt_mask,
                    self.num_classes,
                    pred_bboxes.detach()
                )
            #如果条件满足，将调用 self.static_assigner 方法，该方法用于执行静态目标分配策略。在这个方法中，它会传递一系列参数，包括锚框的位置信息 anchor_points、
            # 锚框的步幅信息 stride_tensor、每个锚框的数量列表 num_anchors_list、真实标签信息 gt_labels、真实边界框信息 gt_bboxes、填充掩码信息 pad_gt_mask、
            # 类别数量 self.num_classes 以及预测的边界框信息 pred_bboxes.detach()。这个方法的目标是为每个预测的锚框分配真实目标，返回分配的类别标签 assigned_labels、
            # 边界框 assigned_bboxes 和分配的分数 assigned_scores。
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach(),
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
        alpha_l = -1
        # cls loss
        if self.use_varifocal_loss:#todo 走变焦损失函数
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()#码计算了 assigned_scores 中的所有元素的总和，并将结果存储在 assigned_scores_sum 变量中。这可能用于后续的损失计算或其他目的，
        # 具体取决于代码的其他部分。总之，这段代码的主要作用是计算分类损失，并计算了 assigned_scores 的总和。
        if paddle.distributed.get_world_size() > 1:#首先，代码检查当前训练是否在分布式环境中进行。通过 paddle.distributed.get_world_size() 函数，可以获取当前分布式训练的进程数。如果进程数大于1，说明在分布式环境中。
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1.)
        else:
            assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum
#如果不在分布式环境中，即只有一个进程在训练，那么代码仅使用 paddle.clip() 函数将 assigned_scores_sum 限制在一个最小值为1的范围内。
        loss_iou, loss_dfl = self._bbox_loss(pred_angle, pred_bboxes,#这段代码用于计算目标检测中的两个损失项：IoU 损失（Intersection over Union loss）和 DFL 损失（Distance-Focal loss）。下面对代码逐行解释：
                                             anchor_points, assigned_labels,
                                             assigned_bboxes, assigned_scores,
                                             assigned_scores_sum, stride_tensor)
        #self._bbox_loss 是一个函数，用于计算目标检测中的边界框回归损失。该函数接受多个参数
        # pred_angle：预测的边界框角度信息。
        # pred_bboxes：预测的边界框位置信息。
        # anchor_points：锚点位置。
        # assigned_labels：分配给预测框的真实类别标签。
        # assigned_bboxes：分配给预测框的真实边界框位置信息。
        # assigned_scores：分配给预测框的真实得分。
        # assigned_scores_sum：分配给预测框的真实得分之和。
        # stride_tensor：步幅信息。

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl
        }
        return out_dict#误差字典

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    @staticmethod
    def _df_loss(pred_dist, target):#todo dfl函数
        target_left = paddle.cast(target, 'int64')#将目标角度信息 target 强制转换为整数类型，用于计算损失。
        target_right = target_left + 1#target_right = target_left + 1：创建 target_right，表示目标角度信息加1，用于计算损失。
        # 这是因为 DFL 要度量预测角度信息与目标角度信息之间的差异，而 target_left 表示的是目标角度信息的下界，target_right 表示的是目标角度信息的上界。
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left#用于加权损失。weight_left 表示距离目标角度信息左边界的距离，而 weight_right 表示距离目标角度信息右边界的距离。这些权重在计算损失时用于加权。
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(#这两行计算左边界和右边界的交叉熵损失。使用预测的角度分布 pred_dist 和 target_left、target_right 来计算损失
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)#将左边界和右边界的损失相加，并取均值，最终返回一个标量值作为DFL损失的结果。这个损失度量了预测的
        # 角度信息与目标角度信息之间的差异，通过左边界和右边界的加权交叉熵来实现。

    def _bbox_loss(self, pred_angle, pred_bboxes, anchor_points,
                   assigned_labels, assigned_bboxes, assigned_scores,
                   assigned_scores_sum, stride_tensor):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)#num_classes=15,mask_positive 中的元素为 True 表示对应位置的预测框的标签不是背景类别，而为目标类别
        num_pos = mask_positive.sum()#num_pos 是一个整数，它表示非背景类别的预测框的数量。这个值是通过对 mask_positive 进行求和操作得到的，即统计了 True 值的数量，即非背景类别的预测框数量。
        # pos/neg loss
        if num_pos > 0:#todo 走
            # iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 5])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 5])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 5])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).reshape([-1])

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            # todo dfl计算角度损失
            angle_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, self.angle_max + 1])
            pred_angle_pos = paddle.masked_select(
                pred_angle, angle_mask).reshape([-1, self.angle_max + 1])
            assigned_angle_pos = (
                assigned_bboxes_pos[:, 4] /
                self.half_pi_bin).clip(0, self.angle_max - 0.01)
            loss_dfl = self._df_loss(pred_angle_pos, assigned_angle_pos)
        else:
            loss_iou = pred_bboxes.sum() * 0.
            loss_dfl = paddle.zeros([1])

        return loss_iou, loss_dfl
#############################这段代码的作用是将输入的边界框从中心-宽度-高度-角度的表示方式（x, y, w, h, angle）
    # 转换为角点的表示方式（x1, y1, x2, y2, x3, y3, x4, y4），其中角点表示方式是由四个角点的坐标组成
    def _box2corners(self, pred_bboxes):
        """ convert (x, y, w, h, angle) to (x1, y1, x2, y2, x3, y3, x4, y4)

        Args:
            pred_bboxes (Tensor): [B, N, 5]
        
        Returns:
            polys (Tensor): [B, N, 8]
        """
        x, y, w, h, angle = paddle.split(pred_bboxes, 5, axis=-1)
        cos_a_half = paddle.cos(angle) * 0.5
        sin_a_half = paddle.sin(angle) * 0.5
        w_x = cos_a_half * w
        w_y = sin_a_half * w
        h_x = -sin_a_half * h
        h_y = cos_a_half * h
        return paddle.concat(
            [
                x + w_x + h_x, y + w_y + h_y, x - w_x + h_x, y - w_y + h_y,
                x - w_x - h_x, y - w_y - h_y, x + w_x - h_x, y + w_y - h_y
            ],
            axis=-1)
##############################这段代码是目标检测模型的后处理步骤，用于从模型的输出中生成最终的检测框和检测结果。以下是代码的逐行解释：
    def post_process(self, head_outs, scale_factor):
        pred_scores, pred_bboxes = head_outs#todo 到目前预测框的参数是带角度的：从模型的输出 head_outs 中解析出预测的分数和边界框。
        # [B, N, 5] -> [B, N, 8]
        pred_bboxes = self._box2corners(pred_bboxes)#这段代码的作用是将输入的边界框从中心-宽度-高度-角度的表示方式（x, y, w, h, angle）转换为角点的表示方式（x1, y1, x2, y2, x3, y3, x4, y4），其中角点表示方式是由四个角点的坐标组成
        # scale bbox to origin
        scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)#将尺度因子 scale_factor 沿着最后一个轴（axis=-1）分割成两个部分，分别表示垂直方向和水平方向的尺度因子。
        scale_factor = paddle.concat(
            [
                scale_x, scale_y, scale_x, scale_y, scale_x, scale_y, scale_x,
                scale_y
            ],
            #将尺度因子重复拼接，以适应每个角点的坐标。具体来说，将两个尺度因子分别重复4次，然后在最后一个轴上进行拼接，
            # 最终得到一个形状为 [B, N, 8] 的张量，其中 B 表示批量大小，N 表示边界框的数量，8 表示每个边界框由8个坐标值组成。
            axis=-1).reshape([-1, 1, 8])
        pred_bboxes /= scale_factor#将预测的边界框坐标除以尺度因子，以将边界框从缩放后的图像尺寸还原到原始图像尺寸
        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        return bbox_pred, bbox_num#用非极大值抑制（NMS）方法对还原后的边界框进行过滤，以去除重叠的边界框并选择得分最高的边界框。bbox_pred 包含了NMS后的边界框坐标，bbox_num 表示保留的边界框数量。
