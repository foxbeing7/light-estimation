import argparse
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import LightSourceEstimationModel
from loaddata import load_dataset, train_loader, val_loader, test_loader , CustomDataset, transform
from losses import CustomLoss
import pandas as pd
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Light Source Estimation')
parser.add_argument('--mode', type=str, choices=['train', 'test'], default= 'train', help='Choose mode: train or test')
parser.add_argument("--test_image_dir", type=str,default="./testset/imgs",help='the dir of testset')
# parser.add_argument("--num_epoch", type=int, default=10 , help='the epoch of train')
parser.add_argument("--save_dir", type=str, default='./checkpoint/' , help='the path of saved model')
parser.add_argument("--resume", type=str, default=None , help='the path of resume model')
args = parser.parse_args()

num_epoch = 100
batch_size = 32
learning_rate = 0.001
num_outputs = 4  # x, y, z, yaw
eval_interval = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_file = './checkpoint/log4.txt'
model = LightSourceEstimationModel(num_outputs).to(device)

criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if args.mode == 'train':

    if args.resume is not None:
        print('loading trained model...')
        model.load_state_dict(torch.load(args.resume))


    print('-----------------------------------training process-----------------------------------------')

    best_val_loss = float('inf')

    for epoch in range(num_epoch):

        model.train()
        loss_df = pd.DataFrame(columns=['Epoch','Loss'])
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
        # for batch in train_loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs.to(torch.float32)
            labels = labels.to(torch.float32)
            loss = criterion(outputs, labels)

            # position_loss.backward()
            loss.backward()
            optimizer.step()

            if (i + 1) % eval_interval == 0 :

                with open(log_file, 'a') as f :
                    f.write(f"Epoch: {epoch+1},Total Loss: {loss.item()}\n")
                loss_df = loss_df.append({'Epoch': epoch + 1, 'Loss': loss.item()}, ignore_index=True)
                # loss_df.to_csv('./checkpoint/logs/loss_log.csv',index = False)
            running_loss += loss.item()

            loss_df.to_csv('./checkpoint/logs/loss_log8.csv', index=False, mode='a', header=False)
            print(f"Epoch [{epoch + 1}/{num_epoch}] - Batch [{i + 1}/{len(train_loader)}] - Train Loss: {loss.item()}")

        if (epoch + 1) % eval_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch['image'].to(device), batch['label'].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epoch}] - Validation Loss: {val_loss / len(val_loader)}")

            if val_loss / len(val_loader) < best_val_loss:
                best_val_loss = val_loss / len(val_loader)
                torch.save(model.state_dict(), os.path.join(args.save_dir,'bestV9.pth'))
                print(f"best Model saved at {args.save_dir}/best.pth")
                print('-----------------------------------------------------------------------------')

    torch.save(model.state_dict(), args.save_dir+"latestV9.pth")
    print(f"lastest Model saved at {args.save_dir}")



    print('-----------------------------------test process-----------------------------------------')
    print('loading model')
    model.load_state_dict(torch.load(os.path.join(args.save_dir,'bestV9.pth')))
    model.eval()

    predictions = []
    image_names = []
    actual_labels = []
    test_loss = 0.0
    # print('loading data...this process will take a few minutes')
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        for batch in test_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predictions.extend(outputs.cpu().numpy())
            image_names.extend(batch['filename'])
            actual_labels.extend(labels.cpu().numpy())

    num_test_samples = len(test_loader.dataset)
    average_test_loss = test_loss / num_test_samples

    predictions_df = pd.DataFrame(predictions, columns=['pre_x', 'pre_y', 'pre_z', 'pre_yaw'])
    actual_labels_df = pd.DataFrame(actual_labels, columns=['x', 'y', 'z', 'yaw'])
    image_names_df = pd.DataFrame({'image_name': image_names})
    results_df = pd.concat([image_names_df, predictions_df, actual_labels_df], axis=1)

    results_df.to_csv("./test_results/test_resultsV9.csv", index=False , mode='a',header=not os.path.exists("./test_results/test_resultsV9.csv"))
    print(f"test result saved at ./test_results")
    print(f"Average Test Loss: {average_test_loss}")
elif args.mode == 'test':
    print('-----------------------------------test process-----------------------------------------')
    print('loading model')
    model.load_state_dict(torch.load(os.path.join(args.save_dir,'bestV5.pth')))
    model.eval()

    predictions = []
    image_names = []
    actual_labels = []
    test_loss = 0.0
    # print('loading data...this process will take a few minutes')
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        for batch in test_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predictions.extend(outputs.cpu().numpy())
            image_names.extend(batch['filename'])
            actual_labels.extend(labels.cpu().numpy())

    num_test_samples = len(test_loader.dataset)
    average_test_loss = test_loss / num_test_samples

    predictions_df = pd.DataFrame(predictions, columns=['pre_x', 'pre_y', 'pre_z', 'pre_yaw'])
    actual_labels_df = pd.DataFrame(actual_labels, columns=['x', 'y', 'z', 'yaw'])
    image_names_df = pd.DataFrame({'image_name': image_names})
    results_df = pd.concat([image_names_df, predictions_df, actual_labels_df], axis=1)

    results_df.to_csv("./test_results/test_resultsV5.csv", index=False , mode='a',header=not os.path.exists("./test_results/test_resultsV5.csv"))
    print(f"test result saved at ./test_results")
    print(f"Average Test Loss: {average_test_loss}")

# test
else:
    print("Invalid mode. Please choose either 'train' or 'test'.")