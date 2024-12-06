from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn




class SimpleMLP(nn.Module):
    def __init__(self, input_size=230, hidden_size=128, output_size=6):
        super(SimpleMLP, self).__init__()
        self.re_embedding_layer = nn.Linear(768, 1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x.shape = (1, 230, 768)
        # print(f"shape of x is {x.shape}")
        x = self.re_embedding_layer(x.permute(1, 0, 2))
        x = x.squeeze(2)
        # print(f"shape of x is {x.shape}")
        x = torch.relu(self.fc1(x))
        # print(f"shape of x is {x.shape}")
        x = self.fc2(x)
        # print(f"shape of x is {x.shape}")
        return x

# 定义一个简单的 RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)  # RNN层
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层，用于输出

    def forward(self, x):
        output, hidden = self.rnn(x)
        output = self.fc(output[-1])
        return output

# 定义一个简单的 GRU
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size)  # GRU层
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, x):
        output, hidden = self.gru(x)
        output = self.fc(output[-1])
        return output
    

# 定义一个简单的 LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)  # LSTM层
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        output = self.fc(output[-1])
        return output

class StateEmbedding(nn.Module):
    def __init__(self, input_size=768, hidden_size=1):
        super(StateEmbedding, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
     
        return x


class CharaNet(nn.Module):
    def __init__(self, input_size=128, chara_embeding_size = 200):
        super(CharaNet, self).__init__()
        self.RNN = nn.RNN(input_size, chara_embeding_size)
        # input shape of self.RNN is (input_seq_len, batch_size, input_size)     [windowsize, 1, 230]
        #         # output shape of self.RNN is [windowsize, 1, chara_embedding_size]

    def forward(self, x):
        # shape of x is [bs, windowsize, 203]
        # print(f"shape of x here is  {x.shape}")
        x = x.permute(1, 0, 2)
        _, hidden = self.RNN(x)
        # shape of hidden is [1, 1, chara_embedding_size]
        
        return hidden
    
class MentNet(nn.Module):
    def __init__(self, len_state_embedding, chara_embedding_size, output_size=6):
        super(MentNet, self).__init__()
        input_szie = len_state_embedding + chara_embedding_size

        self.fc1 = nn.Linear(input_szie, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()



    def forward(self, x, chara_embedding):
        # shape of X is {bs, len} = {1, 200}
        # shaoe of character_embedding is [1, 1, chara_embedding_size]
        input_x = torch.cat((x, chara_embedding.squeeze(0)), dim=1)
        x = torch.relu(self.fc1(input_x))
        
        self.mental_state = x

        x = self.fc2(x)

        return x
    
class SortingHat(nn.Module):
    def __init__(self, window_size = 3, chara_embedding_size = 200, hidden_size=128, output_size=6, len_state_embedding = 200, len_action_embedding=3):
        super(SortingHat, self).__init__()
        self.statembed_net = StateEmbedding(input_size=768, hidden_size=1)
        character_len = len_state_embedding + len_action_embedding
        self.chara_net = CharaNet(input_size=character_len, chara_embeding_size = chara_embedding_size)
        self.ment_net = MentNet(len_state_embedding, chara_embedding_size, output_size=output_size)

    def forward(self, trajectory, x):
        # state_embed: [1, 200, 1]
        state_embed = self.statembed_net(x)
        state_embed = state_embed.squeeze(2)

        # trajectory_embed: [1, window_size, 203, 1]
        trajectory_embed = self.statembed_net(trajectory)

        trajectory_embed = trajectory_embed.squeeze(3)

        character_embedding = self.chara_net(trajectory_embed)
        self.chara = character_embedding

        output = self.ment_net(state_embed, character_embedding)

        return output





if __name__ == '__main__':
    test_rnn()
    print(f"{'='*20}")
 

    # we need to build up a model for handling the embedded dataset. 

    