import torch.nn.functional as F
import torch.nn as nn
import math

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, prediction, target):

        # pred_position = prediction[:, :3]  # x, y, z
        #
        # pred_yaw = prediction[:, 3]  # yaw
        pred_position = prediction[:, :3]  # x, y, z
        pred_yaw = prediction[:, 3]  # yaw

        target_position = target[:, :3]
        target_yaw = target[:, 3]

        # 计算位置的平滑 L1 损失
        position_loss_x = F.mse_loss(pred_position[:, 0], target_position[:, 0])
        position_loss_y = F.mse_loss(pred_position[:, 1], target_position[:, 1])
        position_loss_z = F.mse_loss(pred_position[:, 2], target_position[:, 2])
        # target_position = target[:, :3]
        # target_yaw = target[:, 3]
        # position_loss = F.mse_loss(pred_position, target_position)
        # position_loss = math.sqrt((pred_position[:, 0]-target_position[:, 0])**2+(pred_position[:, 1]-target_position[:, 1])**2+(pred_position[:, 2]-target_position[:, 2])**2)
        position_loss =  position_loss_x + position_loss_y + position_loss_z
        yaw_loss = F.smooth_l1_loss(pred_yaw, target_yaw)

        total_loss = 0.5 * position_loss + 0.5 * yaw_loss

        return total_loss


