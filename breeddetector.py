#!/usr/bin/python3
import jetson_inference
import jetson_utils
from jetson_utils import cudaCrop,cudaDeviceSynchronize,cudaAllocMapped, saveImage
import argparse
import time
import os

parser = argparse.ArgumentParser(description="Find the breed of dog in an image",formatter_class=argparse.RawTextHelpFormatter,epilog="uh idk what to put here")
parser.add_argument("--headed", "-w",type=bool,default=False,help="True/False, If the camera window should be opened. Defaults to False")
parser.add_argument("--threashold","-t", type=float, default=.9,help="Percent needed for species identification. Use 0 to disable. Defaults to 0.9")
parser.add_argument("--device","-d", type=str,default="/dev/video0",help="Camera/RTP Stream to use. Defaults to /dev/video0 (usb camera). Use the command ls /dev/video* to check devices")
parser.add_argument("--filename","-f", type=str,default="dog",help="The filename to use for the image and videos (use a path to save outside of the current directory). DO NOT INCLUDE A FILE EXTENSION")
args = parser.parse_known_args()[0]
headed = bool(args.headed)
if args.threashold > 1:
    percentneeded = float(args.threashold)/100
else:
    percentneeded = float(args.threashold)
filename = str(args.filename).replace(".mp4","").replace(".jpg","")

#Instanciate Detection Objects
starttime = time.time() #get the time before loading the objectnet
video_net = jetson_inference.detectNet("coco-dog",['--log-level=error'],threshold=0.5) #load the objectnet
print(f"Time spent to Load Video: {int(time.time()-starttime)} seconds") #find loading duration

# Define the video Source.
video_camera = jetson_utils.videoSource(args.device) #define the video camera with -d

if headed:
    display = jetson_utils.videoOutput(f"file://{args.filename}.mp4") #create file to stream to window
else:
    print("program is running in headless mode")
#Instanciate Image Recognition Objects
starttime = time.time() #get time before loading the imagenet
image_net = jetson_inference.imageNet('googlenet',['--log-level=error']) #load the aforementioned imagenet
print(f"Time spent to Create Network: {int(time.time()-starttime)} seconds") #get time after loading the aforementioned aforementioned imagenet, and subtract the aforementioned time before loading the aforementioned aforementioned aforementioned imagenet

#its loop time
dogsdetected = 0
print("Program starting!")
try:
    while True:
        img = video_camera.Capture() #capture frame
        detections = video_net.Detect(img) #start detection
        if headed:
            display.Render(img) #create/update window
            display.SetStatus("Object Detection | Network {:.0f} FPS".format(video_net.GetNetworkFPS())) #set title
        else:
            pass
        ### Dog detection time (dog/cat/person/etc)
        #Get ClassID for the objects
        for detection in detections: #for every detection
            class_idx = detection.ClassID #object types
            class_confidence = detection.Confidence #confidence
            class_desc = video_net.GetClassDesc(class_idx) #descriptions for every class (i.e. Dog), breed identification coming soon
            #print(f"{class_desc} at {class_confidence*100}%") #uncomment this if you want to hear about everything detected (prints "dog at 90%")

           ###Breed identification time (i.e. poodle, husky)
            if class_desc =="dog" and class_confidence >= int(percentneeded): #if its confident that we found a dog
                print(f"Dog found with {class_confidence * 100}% confidence") #ok we found a dog, time to classify it
                roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)) #dimensions of boundary box
                snapshot = cudaAllocMapped(width=roi[2]-roi[0], height=roi[3]-roi[1], format=img.format) #take photo
                cudaCrop(img, snapshot, roi) #crop photo
                cudaDeviceSynchronize()
                saveImage(f"{args.filename}.jpg",snapshot) #save the image in the directory specified with -d
                del snapshot
                dog_img = jetson_utils.loadImage("dog.jpg") #gotta reload that image, man
                dog_idx, confidence = image_net.Classify(dog_img) #classify dog selection
                dog_class_desc = image_net.GetClassDesc(dog_idx) #breed class ids
                print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(dog_class_desc, dog_idx, confidence * 100)) #print statement
                dogsdetected+=1
except Exception as e: #if code is stopped
    if isinstance(e,KeyboardInterrupt) or isinstance(e,InterruptedError) or e == KeyboardInterrupt or e == InterruptedError: #if Ctrl+C is used
        print("Stopping!")
        print(f"You found {dogsdetected} dogs!") #cool dog count
    else:
        print(f"a {type(e)} occured, with the error being {e}")