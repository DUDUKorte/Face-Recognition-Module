# Face-Recognition-Module
Main Face Recognition Module
---
# Pre-Requirements
##  - `The libraries in requirements.txt`
##  - `dlib With or without CUDA enabled (we recommend enable CUDA support)`
---
# Instaling dlib on Windows with CUDA enabled and disabled
## **1** Instaling without CUDA Support
###  - `pip install CMake`
###  - `pip install dlib`
## **2** Instaling with CUDA Support
## **2.1** Install these pre-requirements:
### - `CUDA ToolKit`
### - `CMake` 
### - `Anaconda3 or miniconda3`
### - `cudnn (just copy cudnn inside /anaconda3/ and /anaconda3/bin/)`
### - `dlib from github`
### - `Visual Studio and C++ Build Tools`
### - `Anaconda3 or miniconda3`
## **2.2** Unzip dlib folder anywhere
## **2.3** Create "build" folder inside dlib folder (path/to/your/dlib/build/)
## **2.4** Configure dlib with CMake
### - `Open CMake GUI`
### - `Set: where is the source code: /path/to/your/dlib`
### - `Set: wheres to build the binaries: /path/to/your/dlib/build`
### - `Click "Configure" Button and just press oK or Continue`
### - `When finish the process will apear the dlib's variables`
### - `Seacrh for DLIB_USE_CUDA and check the box or set to "ON"`
### - `Click "Configure" and wait, when its done, should apear "DLIB WILL USE CUDA"`
### ![image](https://github.com/DUDUKorte/Face-Recognition-Module/assets/40546705/bc7f5bb8-0187-4a2e-80ba-9ff6406f60b1)

## **2.5** Install dlib in conda
### - `Go to dlib's folder with anaconda terminal (like ```conda cd your/path/to/dlib``` )`
### - `Execute ```python setup.py install``` `
### - `Will install dlib in your conda, this will dont work if you dont have cudnn inside conda`
### - `Like before, should apear "DLIB WILL USE CUDA"`
### - `DONT FORGET: TEST DLIB BEFORE USE IT`
### - run a python file with this:
```
import dlib
print(dlib.DLIB_USE_CUDA)
```
### - if you get True, enjoy
---
## **3** Using Face Recognition System DEMO
### - After installing dlib and other libraries from requirements.txt, you just need to run "demoFaceRecognitionSystem.py"
---
