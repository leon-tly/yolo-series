import torch
import torch.nn as nn
import torch.nn.functional as F

class reorg(nn.Module):
    def __init__(self):
        super(reorg, self).__init__()
        self.Conv = nn.Sequential(
            # [Nx32x416x416]
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.Conv(x)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)



class YOLOv2(nn.Module):
    def __init__(self,classes_num=20):
        super(YOLOv2, self).__init__()
        self.classes_num = classes_num
        # 输入 [Nx3x416x416]
        self.Conv1_32 = nn.Sequential(
            # [Nx32x416x416]
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        # [N*32*208*208]
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2_64 = nn.Sequential(
            # [Nx64x208x208]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        # [Nx64x104x104]
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv3_128 = nn.Sequential(
            # [Nx128x104x104]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.Conv4_64 = nn.Sequential(
            # [Nx64x104x104]
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.Conv5_128 = nn.Sequential(
            # [Nx128x104x104]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
         # [Nx128x52x52]
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv6_256 = nn.Sequential(
            # [Nx256x52x52]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.Conv7_128 = nn.Sequential(
            # [Nx128x52x52]
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.Conv8_256 = nn.Sequential(
            # [Nx256x52x52]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        # [Nx256x26x26]
        self.MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv9_512 = nn.Sequential(
            # [Nx512x26x26]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.Conv10_256 = nn.Sequential(
            # [Nx256x26x26]
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.Conv11_512 = nn.Sequential(
            # [Nx512x26x26]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.Conv12_256 = nn.Sequential(
            # [Nx256x26x26]
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.Conv13_512 = nn.Sequential(
            # [Nx512x26x26]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        ###########
        # contact #
        ###########

        # [Nx512x13x13]
        self.MaxPool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv14_1024 = nn.Sequential(
            # [Nx1024x13x13]
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        self.Conv15_512 = nn.Sequential(
            # [Nx512x13x13]
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.Conv16_1024 = nn.Sequential(
            # [Nx1024x13x13]
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        self.Conv17_512 = nn.Sequential(
            # [Nx512x13x13]
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.Conv18_1024 = nn.Sequential(
            # [Nx1024x13x13]
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        self.Conv19_1024 = nn.Sequential(
            # [Nx1024x13x13]
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        self.Conv20_1024 = nn.Sequential(
            # [Nx1024x13x13]
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )

        ###########
        # contact #
        ###########

        ###########
        # coding
        # 1280
        # 512 -> conv(512, 64, 1) -> 64
        # 64x26x26 -> reorg -> 256x13x13
        # 256, 1024 -> cat -> 1280
        ###########

        self.Conv21_1024 = nn.Sequential(
            # [Nx1024x13x13]
            nn.Conv2d(1280, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )

        self.Conv22_125 = nn.Sequential(
            # [Nx125x13x13]
            # 125 = (5+20)*5   5:x,y,w,h,conf  20:classes  5:anchor num
            nn.Conv2d(1024, (5+self.classes_num)*5, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.Linear()
        )

        self.reorg_layer = reorg()

    def forward(self, x):
        x = self.Conv1_32(x)
        x = self.MaxPool1(x)
        x = self.Conv2_64(x)
        x = self.MaxPool2(x)
        x = self.Conv3_128(x)
        x = self.Conv4_64(x)
        x = self.Conv5_128(x)
        x = self.MaxPool3(x)
        x = self.Conv6_256(x)
        x = self.Conv7_128(x)
        x = self.Conv8_256(x)
        x = self.MaxPool4(x)
        x = self.Conv9_512(x)
        x = self.Conv10_256(x)
        x = self.Conv11_512(x)
        x = self.Conv12_256(x)
        x = self.Conv13_512(x)
        y = x ## contact
        x = self.MaxPool5(x)
        x = self.Conv14_1024(x)
        x = self.Conv15_512(x)
        x = self.Conv16_1024(x)
        x = self.Conv17_512(x)
        x = self.Conv18_1024(x)
        x = self.Conv19_1024(x)
        x = self.Conv20_1024(x)
        y = self.reorg_layer(y)
        x = torch.cat([x, y], dim=1)
        x = self.Conv21_1024(x)
        x = self.Conv22_125(x)
        return x


class YOLOv2Loss(nn.Module):
    def __init__(self, classes_num=20):
        super(YOLOv2Loss, self).__init__()
        self.classes_num = classes_num
        self.anchors_num = 5

    def forward(self, pred):
        bsize, anchorC, H, W = pred.size()
        # 1x13x13x125
        pred = pred.permute(0,2,3,1).contiguous().view(bsize, H*W*self.anchors_num, self.classes_num + 5)
        xy_pred = torch.sigmoid(pred[:, :, 0:2])
        conf_pred = torch.sigmoid(pred[:, :, 4:5])
        hw_pred = torch.exp(pred[:, :, 2:4])
        class_score = pred[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)
        print(delta_pred)
        return delta_pred, class_score, class_pred


def box_ious(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2 (x1, y1, x2, y2)
    Arguments:
    box1 -- tensor of shape (N, 4), first set of boxes
    box2 -- tensor of shape (K, 4), second set of boxes
    Returns:
    ious -- tensor of shape (N, K), ious between boxes
    """

    N = box1.size(0)
    K = box2.size(0)

    # when torch.max() takes tensor of different shape as arguments, it will broadcasting them.
    # [N,1] 广播为 [N,K], [1,K] 广播为 [N,K]
    # N 为预测框的个数 (一个grid cell 5个预测框)
    # K 为 真实目标框的个数
    # 计算每一个预测框和真实目标框中最大的x1,y1,最小的x2,y2
    # 从而算出每一个预测框和真实目标框的相交面积
    xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, K))
    yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, K))
    xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, K))
    yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, K))

    # we want to compare the compare the value with 0 elementwise. However, we can't
    # simply feed int 0, because it will invoke the function torch(max, dim=int) which is not
    # what we want.
    # To feed a tensor 0 of same type and device with box1 and box2
    # we use tensor.new().fill_(0)

    iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))
    ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))

    inter = iw * ih

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, K)

    union_area = box1_area + box2_area - inter

    ious = inter / union_area

    return ious


def generate_all_anchors(anchors, H, W):
    """
    Generate dense anchors given grid defined by (H,W)
    Arguments:
    anchors -- tensor of shape (num_anchors, 2), pre-defined anchors (pw, ph) on each cell
    H -- int, grid height
    W -- int, grid width
    Returns:
    all_anchors -- tensor of shape (H * W * num_anchors, 4) dense grid anchors (c_x, c_y, w, h)
    """

    # number of anchors per cell
    A = anchors.size(0)

    # number of cells
    K = H * W

    shift_x, shift_y = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)])

    # transpose shift_x and shift_y because we want our anchors to be organized in H x W order
    shift_x = shift_x.t().contiguous()
    shift_y = shift_y.t().contiguous()

    # shift_x is a long tensor, c_x is a float tensor
    c_x = shift_x.float()
    c_y = shift_y.float()

    centers = torch.cat([c_x.view(-1, 1), c_y.view(-1, 1)], dim=-1)  # tensor of shape (h * w, 2), (cx, cy)

    # add anchors width and height to centers
    all_anchors = torch.cat([centers.view(K, 1, 2).expand(K, A, 2),
                             anchors.view(1, A, 2).expand(K, A, 2)], dim=-1)

    all_anchors = all_anchors.view(-1, 4)

    return all_anchors

def xywh2xxyy(box):
    """
    Convert the box encoding format form (c_x, c_y, w, h) to (x1, y1, x2, y2)
    Arguments:
    box -- tensor of shape (N, 4), box of (c_x, c_y, w, h) format
    Returns:
    xxyy_box -- tensor of shape (N, 4), box of (x1, y1, x2, y2) format
    N = H*W*num_anchor
    """

    x1 = box[:, 0] - (box[:, 2]) / 2
    y1 = box[:, 1] - (box[:, 3]) / 2
    x2 = box[:, 0] + (box[:, 2]) / 2
    y2 = box[:, 1] + (box[:, 3]) / 2

    x1 = x1.view(-1, 1)
    y1 = y1.view(-1, 1)
    x2 = x2.view(-1, 1)
    y2 = y2.view(-1, 1)

    xxyy_box = torch.cat([x1, y1, x2, y2], dim=1)
    return xxyy_box

def box_transform_inv(box, deltas):
    """
    apply deltas to box to generate predicted boxes
    Arguments:
    box -- tensor of shape (N, 4), boxes, (c_x, c_y, w, h)
    deltas -- tensor of shape (N, 4), deltas, (σ(t_x), σ(t_y), exp(t_w), exp(t_h))
    Returns:
    pred_box -- tensor of shape (N, 4), predicted boxes, (c_x, c_y, w, h)
    """

    c_x = box[:, 0] + deltas[:, 0]
    c_y = box[:, 1] + deltas[:, 1]
    w = box[:, 2] * deltas[:, 2]
    h = box[:, 3] * deltas[:, 3]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    pred_box = torch.cat([c_x, c_y, w, h], dim=-1)
    return pred_box


def build_target(pred_data, gt_data, H, W):
    anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
    delta_pred_batch = pred_data[0] # 预测偏移量 [B, H*W*anchor_num, 4]
    conf_pred_batch = pred_data[1] # 预测分数 [B, H*W*anchor_num, 1]
    class_pred_batch = pred_data[2] # 类别预测 [B, H*W*anchor_num, classes_num]

    gt_boxes_batch = gt_data[0] # 真实框
    gt_classes_batch = gt_data[1] # 真实类别
    num_boxes_batch = gt_data[2] # 目标框个数

    bsize = delta_pred_batch[0] # batch size

    num_anchors = 5

    # 生成 [bsize, H * W, num_anchors, 1] 维度的全0矩阵
    iou_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    iou_mask = delta_pred_batch.new_ones((bsize, H * W, num_anchors, 1))
    # 生成 [bsize, H * W, num_anchors, 4] 维度的全0矩阵
    box_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 4))
    box_mask = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    class_target = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    class_mask = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    anchors = torch.FloatTensor(anchors)
    # 生成[13*13*anchor_num，4]维的grid
    # [x,y,w,h] x,y为每个grid cell的左上角坐标
    # w,h为anchors的宽 高
    all_grid_xywh = generate_all_anchors(anchors, H, W)
    # 创建数据类型、device和delta_pred_batch相同的tensor
    all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh)
    all_anchors_xywh = all_grid_xywh.clone()
    # 将左上角坐标+0.5，变为中心点坐标
    all_anchors_xywh[:,0:2] += 0.5
    all_anchors_xxyy = xywh2xxyy(all_anchors_xywh)

    # 遍历batch
    for b in range(bsize):
        # 拿到当前图片的objs数量
        num_obj = num_boxes_batch[b].item()
        # 拿到当前图片的预测的偏移量
        delta_pred = delta_pred_batch[b]

        # 真实box
        gt_boxes = gt_boxes_batch[b][:num_obj, :]
        # 真实标签
        gt_classes = gt_classes_batch[b][:num_obj]
        # gt_boxes = gt_boxes_batch[b]
        # gt_classes = gt_classes_batch[b]
        # 真实坐标是归一化到0-1之间的，乘以W,转化到feature map上的坐标
        # 可以找到回应的grid cell
        # x1, y1, x2, y2
        # x*W, y*H
        gt_boxes[:,0::2] *= W
        gt_boxes[:,1::2] *= H

        all_anchors_xywh = all_anchors_xywh.view(-1, 4)
        # 将预测偏移量叠加在anchor上，得到预测的x,y,w,h
        box_pred = box_transform_inv(all_grid_xywh, delta_pred)
        # 将预测框转为xyxy形式
        box_pred = xywh2xxyy(box_pred)
        # 计算IoU
        ious = box_ious(box_pred, gt_boxes) # shape: (H * W * num_anchors, num_obj)
        # [H*W, num_anchors, num_obj]
        # H*W个grid_cell，每个grid cell中的每个anchor和真实目标框的iou
        ious = ious.view(-1, num_anchors, num_obj)
        # 找到最大的iou，并且保证输出的维度不变
        max_iou, _ = torch.max(ious, dim=-1,keepdim=True)
        iou_thresh_filter = max_iou.view(-1)
        n_pos = torch.nozero(iou_thresh_filter).numel()
        if n_pos > 0:
            # 将iou最大的anchor对应位置置为0
            iou_mask[b][max_iou >= cfg.thresh] = 0



if __name__ == '__main__':
    net = YOLOv2()
    yolov2_loss = YOLOv2Loss()
    data = torch.rand((1,3,416,416))
    output = net(data)
    print(output)
    # yolov2_loss(output)
    anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
    print(generate_all_anchors(torch.Tensor(anchors), 13, 13).size())
