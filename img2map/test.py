import cv2
import torch
import os
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    # model_type = "MiDaS_small"
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    img_path = './hallway_dataset/images'
    imgs = os.listdir(img_path)
    random.shuffle(imgs)
    output_path = 'output'

    show = 10

    for i in range(show):
        img = cv2.imread(os.path.join(img_path, imgs[i])) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        plt.imshow(img)
        plt.show()
        plt.imshow(output)
        plt.show()
        # plt.savefig(os.path.join(output_path, 'fig{}'.format(i)))