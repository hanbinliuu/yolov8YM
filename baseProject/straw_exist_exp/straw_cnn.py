import os
import shutil

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


def resize_image(src_image, size=(128, 128), bg_color="white"):
    # resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.ANTIALIAS)
    new_image = Image.new("RGB", size, bg_color)
    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
    return new_image


def load_dataset(data_path):
    transformation = transforms.Compose([
        # Randomly augment the image data
        # Random horizontal flip
        transforms.RandomHorizontalFlip(0.5),
        # Random vertical flip
        transforms.RandomVerticalFlip(0.3),
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )

    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # use torch.utils.data.random_split for training/test split
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    return train_loader, test_loader


class Net(nn.Module):

    # Defining the Constructor
    def __init__(self, num_classes=3):
        super(Net, self).__init__()

        # In the init function, we define each layer we will use in our model

        # Our images are RGB, so we have input channels = 3.
        # We will apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

        # A second convolutional layer takes 12 input channels, and generates 24 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.2)

        # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
        # This means that our feature tensors are now 32 x 32, and we've generated 24 of them
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        # In the forward function, pass the data through the layers we defined in the init function

        # Use a ReLU activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))

        # Use a ReLU activation function after layer 2
        x = F.relu(self.pool(self.conv2(x)))

        # Select some features to drop to prevent overfitting (only drop during training)
        x = F.dropout(self.drop(x), training=self.training)

        # Flatten
        x = x.view(-1, 32 * 32 * 24)
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        # Return class probabilities via a log_softmax function
        return torch.log_softmax(x, dim=1)


def training(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)

        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data)

        # Get the loss
        loss_criteria = nn.CrossEntropyLoss()
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Print metrics so we see some progress
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


def tst(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            loss_criteria = nn.CrossEntropyLoss()
            test_loss += loss_criteria(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # return average loss for the epoch
    return avg_loss


def preprocess_image(image_path):
    image = Image.open(image_path)
    resized_image = resize_image(image, size=(128, 128))
    transformed_image = transforms.ToTensor()(resized_image)
    normalized_image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(transformed_image)
    return normalized_image.unsqueeze(0)  # 添加一个维度作为批处理维度


def predict_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()


def predict_and_save_all_images(model, input_folder, output_folder, device):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image_tensor = preprocess_image(image_path)
        predicted_class = predict_image(model, image_tensor, device)
        original_image = Image.open(image_path)
        draw = ImageDraw.Draw(original_image)
        width, height = original_image.size
        text = str(predicted_class)
        font_size = 80
        font = ImageFont.truetype("arial.ttf", font_size)  # You can replace "arial.ttf" with the path to your preferred font file
        text_width, text_height = draw.textsize(text, font=font)
        text_position = ((width - text_width) // 2, (height - text_height) // 2)
        draw.text(text_position, text, font=font, fill='red')
        output_path = os.path.join(output_folder, image_file)
        original_image.save(output_path)


if __name__ == '__main__':
    training_folder_name = r'C:\Users\lhb\Desktop\train'
    train_folder = r'C:\Users\lhb\Desktop\train2'
    device = "cuda"

    classes = sorted(os.listdir(training_folder_name))

    # # resize
    # size = (128, 128)
    # if os.path.exists(train_folder):
    #     shutil.rmtree(train_folder)
    # for root, folders, files in os.walk(training_folder_name):
    #     for sub_folder in folders:
    #         print('processing folder ' + sub_folder)
    #         saveFolder = os.path.join(train_folder, sub_folder)
    #         if not os.path.exists(saveFolder):
    #             os.makedirs(saveFolder)
    #         file_names = os.listdir(os.path.join(root, sub_folder))
    #         for file_name in file_names:
    #             file_path = os.path.join(root, sub_folder, file_name)
    #             image = Image.open(file_path)
    #             resized_image = resize_image(image, size)
    #             saveAs = os.path.join(saveFolder, file_name)
    #             resized_image.save(saveAs)
    #
    # # dataloader
    # train_loader, test_loader = load_dataset(train_folder)
    # batch_size = train_loader.batch_size
    # print("Data loaders ready to read", train_folder)
    #
    # # cnn model
    # model = Net(num_classes=len(classes)).to(device)
    # print(model)
    #
    # # Define the loss function and optimizer
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # loss_criteria = nn.CrossEntropyLoss()
    #
    # epoch_nums = []
    # training_loss = []
    # validation_loss = []
    #
    # epochs = 100
    # print('Training on', device)
    # for epoch in range(1, epochs + 1):
    #     train_loss = training(model, device, train_loader, optimizer, epoch)
    #     test_loss = tst(model, device, test_loader)
    #     epoch_nums.append(epoch)
    #     training_loss.append(train_loss)
    #     validation_loss.append(test_loss)
    #
    # # 可视化
    # truelabels = []
    # predictions = []
    # model.eval()
    # # 保存模型
    # # torch.save(model.state_dict(), 'straw_cnn.pth')
    #
    # # 验证
    # for data, target in test_loader:
    #     for label in target.data.numpy():
    #         truelabels.append(label)
    #     for prediction in model(data.to(device)).cpu().data.numpy().argmax(1):
    #         predictions.append(prediction)
    #
    # cm = confusion_matrix(truelabels, predictions)
    # df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    # plt.figure(figsize=(7, 7))
    # sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    # plt.xlabel("Predicted Shape", fontsize=20)
    # plt.ylabel("True Shape", fontsize=20)
    # plt.show()

    # 加载模型+test其他照片
    model = Net(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load('straw_cnn.pth'))

    # 测试整个文件夹
    output = r"C:\Users\lhb\Desktop\video_test"
    predict_and_save_all_images(model, r"C:\Users\lhb\Desktop\video_to_images", output, device)

    # 测试单张图片
    image_path = r"C:\Users\lhb\Desktop\testing\20231204184833373.jpeg"  # 替换为实际的测试图片路径
    image_tensor = preprocess_image(image_path)
    predicted_class = predict_image(model, image_tensor, device)

