
#!/usr/bin/python3
import jetson_inference
import jetson_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("devicepath", type=str, help="path to the device to proccess")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(opt.network)
class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)
print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))