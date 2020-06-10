import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import mex.personalised_rn.dc_task_generator as tg
import os
import math
import argparse

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-tp", "--test_person", type=str, default='03')
parser.add_argument("-w", "--class_num", type=int, default=7)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=5)
parser.add_argument("-e", "--episode", type=int, default=300)
parser.add_argument("-t", "--test_episode", type=int, default=100)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
args = parser.parse_args()

# Hyper Parameters
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
TEST_PERSON = args.test_person

if torch.cuda.is_available():
    dev = "cuda:x"
else:
    dev = "cpu"
device = torch.device(dev)


def write_results(text):
    file_path = 'dc_few_shot.csv'
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(text + '\n')
    else:
        f = open(file_path, 'w')
        f.write(text + '\n')
    f.close()


class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))

    def forward(self, x):
        out = self.layer1(x.float())
        out = self.layer2(out)
        # out = out.view(out.size(0),-1)
        return out  # 64


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=2, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(448, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main():
    write_results(str(args))
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_character_folders, metatest_character_folders = tg.dc_folders(TEST_PERSON)

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.to(device)
    relation_network.to(device)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0
    test_accuracies = []
    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        task = tg.DCTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS)
        sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False)
        batch_dataloader = tg.get_data_loader(task, num_per_class=task.per_class_num, split="test", shuffle=True)

        # sample datas
        samples, sample_labels = sample_dataloader.__iter__().next()
        print('samples.shape', samples.shape)
        batches, batch_labels = batch_dataloader.__iter__().next()
        print('batches.shape', batches.shape)

        # calculate features
        sample_features = feature_encoder(Variable(samples).to(device))
        print('sample_features.shape', sample_features.shape)
        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, 64, 13, 2)
        print('sample_features.shape', sample_features.shape)
        sample_features = torch.sum(sample_features, 1).squeeze(1)
        print('sample_features.shape', sample_features.shape)
        batch_features = feature_encoder(Variable(batches).to(device))
        print('batch_features.shape', batch_features.shape)

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(task.per_class_num * CLASS_NUM, 1, 1, 1, 1)
        print('sample_features_ext.shape', sample_features_ext.shape)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
        print('batch_features_ext.shape', batch_features_ext.shape)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        print('batch_features_ext.shape', batch_features_ext.shape)

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 64 * 2, 13, 2)
        print('relation_pairs.shape', relation_pairs.shape)
        relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
        print('relations.shape', relations.shape)

        mse = nn.MSELoss().to(device)
        one_hot_labels = Variable(
            torch.zeros(task.per_class_num * CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1, 1), 1)).to(device)
        loss = mse(relations, one_hot_labels)

        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        print("episode:", episode + 1, "loss", loss.item())
        # write_results('episode:'+str(episode + 1)+'loss'+str(loss.item()))

        if (episode + 1) % 5 == 0:

            # test
            print("Testing...")
            total_rewards = 0

            for i in range(TEST_EPISODE):
                task = tg.DCTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS)
                sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
                                                       shuffle=False)
                test_dataloader = tg.get_data_loader(task, num_per_class=task.per_class_num, split="test",
                                                     shuffle=True)

                sample_images, sample_labels = sample_dataloader.__iter__().next()

                test_images, test_labels = test_dataloader.__iter__().next()

                # calculate features
                sample_features = feature_encoder(Variable(sample_images).to(device))
                print('sample_features.shape', sample_features.shape)
                sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, 64, 13, 2)
                print('sample_features.shape', sample_features.shape)
                sample_features = torch.sum(sample_features, 1).squeeze(1)
                print('sample_features.shape', sample_features.shape)
                test_features = feature_encoder(Variable(test_images).to(device))
                print('test_features.shape', test_features.shape)

                # calculate relations
                # each batch sample link to every samples to calculate relations
                # to form a 100x128 matrix for relation network
                sample_features_ext = sample_features.unsqueeze(0).repeat(task.per_class_num * CLASS_NUM, 1, 1, 1, 1)
                print('sample_features_ext.shape', sample_features_ext.shape)
                test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
                print('test_features_ext.shape', test_features_ext.shape)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)
                print('test_features_ext.shape', test_features_ext.shape)

                relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, 64*2, 13, 2)
                print('relation_pairs.shape', relation_pairs.shape)
                relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
                print('relations.shape', relations.shape)
                _, predict_labels = torch.max(relations.data, 1)

                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in
                           range(CLASS_NUM * task.per_class_num)]

                total_rewards += np.sum(rewards)

            test_accuracy = total_rewards / 1.0 / CLASS_NUM / task.per_class_num / TEST_EPISODE

            print("test accuracy:", test_accuracy)
            test_accuracies.append(str(test_accuracy))

    write_results(','.join(test_accuracies))


if __name__ == '__main__':
    main()
