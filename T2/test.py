import torch
from net import *
from nail2 import *
from train import *
from torchvision.utils import save_image
from PIL import Image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def save_output_image(tensor, output_path):
    save_image(tensor, output_path)

def test_model_on_image(model, image_path, output_folder):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
    output_image = output.squeeze(0)
    output_filename = os.path.join(output_folder, 'result_image.png')
    save_output_image(output_image, output_filename)

output_folder = r'D:\pythonProject\T2\archive\out_image'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

test_image_path = r'D:\pythonProject\T2\archive\nails_segmentation\images\4c49b502-e402-11e8-97db-0242ac1c0002.jpg'

test_model_on_image(model, test_image_path, output_folder)
