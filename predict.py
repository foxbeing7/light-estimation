import os
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from model import  LightSourceEstimationModel

### to predict with image

transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LightSourceEstimationModel(num_outputs=4)
model.load_state_dict(torch.load("./checkpoint/bestV3.pth"))
model.to(device)
model.eval()

def predict_images_in_folder(image_folder):
    predicted_data = []

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_folder, filename)

            # 打开并预处理图像
            with Image.open(image_path) as img:
                img = transforms(img.convert('RGB')).unsqueeze(0).to(device)  # 添加批次维度并移到设备上

            # 进行光源预测
            with torch.no_grad():
                predicted_light = model(img).squeeze().cpu().numpy()  # 将预测结果移回CPU并转换为NumPy数组

            predicted_data.append([filename] + predicted_light.tolist())

    return predicted_data

if __name__ == "__main__":
    image_folder = r"C:\Users\Admin\PycharmProjects\LECNN\dataset\test"  # 替换为包含要预测的图像的文件夹的路径
    predicted_data = predict_images_in_folder(image_folder)

    # 将预测结果保存到CSV文件
    columns = ["ImageName", "Predicted_x", "Predicted_y", "Predicted_z", "Predicted_yaw"]
    predicted_df = pd.DataFrame(predicted_data, columns=columns)
    predicted_df.to_csv("./predict_result/predict.csv", index=False)