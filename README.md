# Face-Recognition-Module
Main Face Recognition Module
---
# Pre-Requirements
##  - `The libraries in requirements.txt`
##  - `dlib With or without CUDA enabled (we recommend enable CUDA support)`
---
# Instaling requirements.txt
## If you want to use dlib with CUDA support, use requirements_wo_dlib.txt
## ```pip install -r requirements.txt``` or ```pip install -r requirements_wo_dlib.txt```
---
# Instaling dlib on Windows with CUDA enabled
## **1** Instaling with CUDA Support
## **1.1** Install these pre-requirements:
### - `CUDA ToolKit`
### - `CMake` 
### - `Anaconda3 or miniconda3`
### - `cudnn (just copy cudnn inside /anaconda3/ and /anaconda3/bin/)`
### - `dlib from github`
### - `Visual Studio and C++ Build Tools`
### - `Anaconda3 or miniconda3`
## **1.2** Unzip dlib folder anywhere
## **1.3** Create "build" folder inside dlib folder (path/to/your/dlib/build/)
## **1.4** Configure dlib with CMake
### - `Open CMake GUI`
### - `Set: where is the source code: /path/to/your/dlib`
### - `Set: wheres to build the binaries: /path/to/your/dlib/build`
### - `Click "Configure" Button and just press oK or Continue`
### - `When finish the process will apear the dlib's variables`
### - `Seacrh for DLIB_USE_CUDA and check the box or set to "ON"`
### - `Click "Configure" and wait, when its done, should apear "DLIB WILL USE CUDA"`
### ![image](https://github.com/DUDUKorte/Face-Recognition-Module/assets/40546705/bc7f5bb8-0187-4a2e-80ba-9ff6406f60b1)

## **1.5** Install dlib in conda
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
