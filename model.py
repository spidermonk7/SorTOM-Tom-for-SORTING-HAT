import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class StateEmbedding(nn.Module):
    def __init__(self, input_size=768, hidden_size = 4):
        super(StateEmbedding, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        return x

class CharaNet(nn.Module):
    def __init__(self, input_size=128, chara_embeding_size = 40):
        super(CharaNet, self).__init__()
        hidden_size = chara_embeding_size
        self.RNN = nn.RNN(input_size, hidden_size)


    def forward(self, x):
        x = x.permute(1, 0, 2)
        _, hidden = self.RNN(x)

    
        return hidden
    
class MentNet(nn.Module):
    def __init__(self, len_state_embedding, chara_embedding_size, output_size=6):
        super(MentNet, self).__init__()
        input_szie = len_state_embedding
        self.fc1 = nn.Linear(input_szie, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, output_size)
        self.act = nn.ReLU()


    def forward(self, x, chara_embedding):
        # input_x = torch.cat((x, chara_embedding), dim=1)
        input_x = x + chara_embedding
        x = self.act(self.fc1(input_x))
        x = self.act(self.fc2(x) + x)
        x = self.act(self.fc3(x) + x)
        x = self.fc4(x)

        return x
    
class SortingHat(nn.Module):
    def __init__(self, chara_embedding_size = 768, output_size=6, len_state_embedding = 768):
        super(SortingHat, self).__init__()
        
        self.chara_net = CharaNet(input_size=len_state_embedding, chara_embeding_size=chara_embedding_size)
        self.ment_net = MentNet(len_state_embedding , chara_embedding_size=chara_embedding_size, output_size=output_size)

    def forward(self, trajectory, x):
        # state_embed = self.statembed_net(x)
        state_embed = x
        # trajectory_embed = self.trajectory_state_embed_net(trajectory)
        trajectory_embed = trajectory
        character_embedding = self.chara_net(trajectory_embed)
        self.chara = character_embedding    
        character_embedding = character_embedding.view(1, -1)
        output = self.ment_net(state_embed, character_embedding)

        return output

