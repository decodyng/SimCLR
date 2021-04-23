import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model
import torch.nn.functional as F


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, mode, batch_size):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    print(f"Training with mode {mode}")
    batch = 0
    saved_images = False
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        if not saved_images:
            utils.save_rgb_tensor(pos_1[0], 'results/saved_pos1.png')
            utils.save_rgb_tensor(pos_2[0], 'results/saved_pos2.png')
            saved_images = True
        feature_1, z_i = net(pos_1.uniform_(-2.5, 2.5))
        feature_2, z_j = net(pos_2.uniform_(-2.5, 2.5))

        if mode == 'ilr':

            z_i = z_i.uniform_(-.3, .3)
            z_j = z_j.uniform_(-.3, .3)

            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

            mask = (torch.eye(batch_size) * 1e9).cuda()

            # Similarity of the original images with all other original images in current batch. Return a matrix of NxN.
            logits_aa = torch.matmul(z_i, z_i.T)  # NxN

            # Values on the diagonal line are each image's similarity with itself
            logits_aa = logits_aa - mask
            # Similarity of the augmented images with all other augmented images.
            logits_bb = torch.matmul(z_j, z_j.T)  # NxN
            logits_bb = logits_bb - mask
            # Similarity of original images and augmented images
            logits_ab = torch.matmul(z_i, z_j.T)  # NxN
            logits_ba = torch.matmul(z_j, z_i.T)  # NxN

            avg_self_similarity = logits_ab.diag().mean().item()
            logits_other_sim_mask = ~torch.eye(batch_size, dtype=bool, device=logits_ab.device)
            avg_other_similarity = logits_ab.masked_select(logits_other_sim_mask).mean().item()

            # if batch % 5 == 0:
            print(f"Avg self similarity: {avg_self_similarity}")
            print(f"Avg other similarity: {avg_other_similarity}")
            print()
            print()

            # Each row now contains an image's similarity with the batch's augmented images & original images. This applies
            # to both original and augmented images (hence "symmetric").
            logits_i = torch.cat((logits_ab, logits_aa), 1)  # Nx2N
            logits_j = torch.cat((logits_ba, logits_bb), 1)  # Nx2N
            logits = torch.cat((logits_i, logits_j), axis=0)  # 2Nx2N
            logits /= temperature

            # The values we want to maximize lie on the i-th index of each row i. i.e. the dot product of
            # represent(image_i) and represent(augmented_image_i).
            label = torch.arange(batch_size, dtype=torch.long).cuda()
            labels = torch.cat((label, label), axis=0)

            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
        else:

            # [2*B, D]
            out = torch.cat([z_i, z_j], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

            # compute loss
            pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()


        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        batch += 1
    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--mode', default='ilr', type=str, help="Which loss to use")

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k, mode = args.feature_dim, args.temperature, args.k, args.mode
    batch_size, epochs = args.batch_size, args.epochs

    # data prepare
    train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, mode=mode, batch_size=batch_size)
        results['train_loss'].append(train_loss)
        torch.save(model, f'results/epoch_{epoch}.ckpt')
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
