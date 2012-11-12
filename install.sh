#Download and compile the OpenCV library
mkdir opencv_trunk
cd opencv_trunk
echo "Downloading the OpenCV library from https://github.com/itseez/opencv ..."
git clone git://github.com/Itseez/opencv.git
cd opencv/
git reset --hard 9f29506
mkdir release
cd release
cmake .. -DWITH_FFMPEG=OFF
echo "Compiling the OpenCV library..."
make -j 2
cp OpenCVConfig.cmake ../../..
cd ../../..

#Compile the project
mkdir build
cd build
cmake ..
echo "Compiling the project..."
make -j 2
cd ..

