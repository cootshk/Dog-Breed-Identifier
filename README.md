# Classifier

*Note: the [dusty-nv/jetson-inference](https://www.github.com/dusty-nv/jetson-inference) repository is included. If you already have this setup, skip the first few steps.*

![add image descrition here](direct image link here)

## The Algorithm

Add an explanation of the algorithm and how it works. Make sure to include details about how the code works, what it depends on, and any other relevant info. Add images or other descriptions for your project here. 

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

[View a video explanation here](video link)
