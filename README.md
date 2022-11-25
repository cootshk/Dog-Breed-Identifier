![syberian husky](https://www.akc.org/wp-content/uploads/2017/11/Siberian-Husky-standing-outdoors-in-the-winter.jpg)
# Dog Breed Classifier

*Note: the [dusty-nv/jetson-inference](https://www.github.com/dusty-nv/jetson-inference) repository is included. If you already have this setup, skip the first few steps.*



## The Algorithm

Step One: arguments (python script): use flags if you need them. Flags:
```
-h True or --headed True: shows the camera window (warning: can be very laggy) (default is False)
-t {number} or --threashold {number}: how confident you have to be that there's a dog to determine species (default is 0.9)
-d {path/to/device} or --device {path/to/device}: the camera or RTP/RTSP stream to be used, accepted protocols are csi, file, v4l2, rtp, and rtsp (default is /dev/video0 (usb camera), use ls /dev/video* to check your connected usb devices)
-f {path/to/file} or --filename {path/to/file}: the file/path to file to use for image/video classification (defaults to dog). **DO NOT INCLUDE THE FILE EXTENSION IN THIS ARGUEMENT**
```
Step Two: find dogs in the video: The first thing the program does is loads the video passed in the previous step, and checks for dogs using the COCO-Dogs model.

Step Three: once a dog is found, classify the breed: When the program detects a dog with the probability over the threashold, it will save an image to the filename.jpg, then will classify that image using googlenet.

Step Four: output and repeat: Outputs the specified breed, then repeats back to step 2 

*for a more in-depth explanation, feel free to look at the comments in the python files*

## Running this project
THIS PROGRAM REQUIRES A **JETSON NANO**
1. update and install git and cmake `sudo apt update;sudo apt upgrade;sudo apt install git cmake`
2. install the required python packages `sudo apt install libpython3-dev python3-numpy`
3. clone the repository with `sudo git clone --recursive https://github.com/cootshk/hello-ai-world.git`
4. open the jetson-inference directory `cd jetson-inference`
5. make a "build" directory `sudo mkdir build`
6. create the project with CMake `cd build;cmake ../`
7. Install the Google-net image classification models **You do NOT need to install Pytorch**
8. install the jetson-inference library `make;sudo make install;sudo ldconfig`

[Demonstration](video link)
