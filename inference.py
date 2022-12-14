'''
to load learnt parameters
to make inferences of captured views by Pylon cameras
'''

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.parallel
import torch.nn.functional
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import argparse
import numpy as np
import os
import cv2
from PIL import Image
from pypylon import pylon

from html4vision import Col, imagetable

from models.resnet import *
from models.mvcnn import *

MVCNN = 'mvcnn'
RESNET = 'resnet'
MODELS = [RESNET,MVCNN]

parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
parser.add_argument('--num_classes', type=int, help='number of classes')
parser.add_argument('--label', type=int, help='label of tested')
parser.add_argument('--data', metavar='DIR', default='test', help='path to dataset')
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152], type=int, metavar='N', default=34, help='resnet depth (default: resnet34)')
parser.add_argument('--model', '-m', metavar='MODEL', default=RESNET, choices=MODELS,
                    help='pretrained model: ' + ' | '.join(MODELS) + ' (default: {})'.format(RESNET))
parser.add_argument('-c', '--checkpoint', default='checkpoint/resnet34_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

args = parser.parse_args()

print('Loading data')

transform = transforms.Compose([
    transforms.CenterCrop(1200),
    transforms.Resize(224),
    transforms.ToTensor(),
])

def image_loader(data_transforms, image, rotate_angle): 
    image = Image.fromarray(image).convert('RGB')
    tensor_image = data_transforms(image).float()
    tensor_image = tensor_image.unsqueeze(0)
    return tensor_image

def get_inputs(image1,image2,image3):
    #convert images to tensors
    tensor1=image_loader(transform,image1,0) 
    tensor2=image_loader(transform,image2,0) 
    tensor3=image_loader(transform,image3,0)

    inputs = np.stack([tensor1, tensor2, tensor3], axis=1)
    return torch.from_numpy(inputs)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = args.num_classes
label = args.label # actual label

if args.model == RESNET:
    if args.depth == 18:
        model = resnet18(pretrained=args.pretrained, num_classes=num_classes)
    elif args.depth == 34:
        model = resnet34(pretrained=args.pretrained, num_classes=num_classes)
    elif args.depth == 50:
        model = resnet50(pretrained=args.pretrained, num_classes=num_classes)
    elif args.depth == 101:
        model = resnet101(pretrained=args.pretrained, num_classes=num_classes)
    elif args.depth == 152:
        model = resnet152(pretrained=args.pretrained, num_classes=num_classes)
    else:
        raise Exception('Specify number of layers for resnet in command line. --resnet N')
    print('Using ' + args.model + str(args.depth))
else:
    model = mvcnn(pretrained=args.pretrained,num_classes=num_classes)
    print('Using ' + args.model)

model.to(device)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

print('Running on ' + str(device))


# Helper functions
def load_checkpoint():
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(args.checkpoint), 'Error: no checkpoint file found!'

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


load_checkpoint()

device = torch.device("cuda")

# Connect to cameras (Pylon)
tlFactory = pylon.TlFactory.GetInstance()

devices = tlFactory.EnumerateDevices()
if len(devices) == 0:
    raise pylon.RUNTIME_EXCEPTION("No camera present.")

num_cameras = 3

cameras = pylon.InstantCameraArray(min(len(devices), num_cameras))

l = cameras.GetSize()

cam_ip_dict = {}

ip1 = '' 
ip2 = ''
ip3 = ''

ip_order = [ip1, ip2, ip3]

# Create and attach all Pylon Devices.
for i, cam in enumerate(cameras):

    cam.Attach(tlFactory.CreateDevice(devices[i]))
    ip = cam.GetDeviceInfo().GetIpAddress()

    # Print the model name of the camera.
    print("Using device ", cam.GetDeviceInfo().GetModelName(), ip)

    cam_ip_dict[ip] = cam

    cam.Open() 

    cam.GainAuto.SetValue("Off")
    cam.GainRaw.SetValue(2) 
    
    cam.ExposureAuto.SetValue("Off") 
    cam.ExposureTimeRaw.SetValue(2000) 

for ip in ip_order:
    if ip not in [k for k in cam_ip_dict.keys()]:
        raise Exception('problematic ip order')

cameras.StartGrabbing()
converter = pylon.ImageFormatConverter()

converter.OutputPixelFormat = pylon.PixelType_Mono8
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# to save results visually
image_save_count = 0  
img_num_to_save = 200

correctly_labeled = 0
incorrectly_labeled = 0

total = 0.0
correct = 0.0

total_loss = 0.0
n = 0

avg_test_acc = 0
avg_loss = 0


while True:
    with torch.no_grad():
        
        image_arr = []

        for ip in ip_order:
            camera = cam_ip_dict[ip]
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                image_arr.append(image.GetArray())

            
        img1_ = image_arr[0]
        img2_ = image_arr[1]
        img3_ = image_arr[2]

        img1 = img1_[0:1200, 0:1200]
        img2 = img2_[0:1200, 0:1200]
        img3 = img3_[0:1200, 720:1920]     

        inputs = get_inputs(img1, img2, img3)
        inputs = inputs.cuda(device)
        inputs = Variable(inputs)
        outputs = model(inputs)
        targets = torch.tensor([int(label)])

        targets = targets.cuda(device)
        targets = Variable(targets)

        loss = criterion(outputs, targets)

        total_loss += loss
        n += 1

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted.cpu() == targets.cpu()).sum()

        avg_test_acc = 100 * correct / total
        avg_loss = total_loss / n

        predicted_str = str(int(predicted))

        if predicted_str == label:
            correctly_labeled += 1
        else:
            incorrectly_labeled +=1

        outputs = torch.nn.functional.softmax(outputs,dim=1)

        cv2.imwrite("demo_images/view1/view1_" + str(image_save_count) + ".jpg", cv2.resize(img1,(len(img1[0])//4,len(img1)//4)))
        cv2.imwrite("demo_images/view2/view2_" + str(image_save_count) + ".jpg", cv2.resize(img2,(len(img2[0])//4,len(img2)//4)))
        cv2.imwrite("demo_images/view3/view3_" + str(image_save_count) + ".jpg", cv2.resize(img3,(len(img3[0])//4,len(img3)//4)))

        cv2.imwrite("demo_images/result1/result1_" + str(image_save_count) + ".jpg", cv2.resize(top1,(len(top1[0])//2,len(top1)//2)))
        
        if image_save_count > img_num_to_save:
            break

        image_save_count += 1

# table description
cols = [
    Col('id1', '#'), # 1-based indexing
    Col("img", "view 1", "demo_images/view1/view1_*.jpg"),             # make a column of 1-based indices
    Col("img", "view 2", "demo_images/view2/view2_*.jpg"),             # specify image content for column 2
    Col("img", "view 3", "demo_images/view3/view3_*.jpg"),     # specify image content for column 3
    Col("img", "prediction", "demo_images/result1/result1_*.jpg"), # specify image content for column 4
]

html_file_name = 'results.html'

imagetable(cols, html_file_name, 'Results',
    sortable=True,              # enable interactive sorting
    sticky_header=True,         # keep the header on the top
    sort_style='materialize',   # use the theme "materialize" from jquery.tablesorter
    zebra=True,                 # use zebra-striped table
)
