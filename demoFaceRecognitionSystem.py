# TCC - Reconhecimento Facial Módulo Principal DEMO - JULHO 2023
# 
# DEMO v1.0
#
# Esta é uma DEMO do módulo principal de reconhecimento facial do TCC de Eduardo Korte e
# Rordrigo Avelar.
# Instale os pacotes do arquivo requirements.txt que pode ser encontrado junto ao código
# fonte no github oficial em
#
#      http://www.github.com/DUDUKorte
# 
# Esta Demo deve ser executada diretamente do arquivo principal "main.py"
# para funcionar corretamente.
# 
# ESTA É UMA DEMO QUE AINDA ESTÁ EM DESENVOLVIMENTO. PROBLEMAS PODEM OCORRER
# Favor, relatar problemas no github do projeto.
#=======================================================================================
# TCC - Facial Recognition Main Module DEMO - JULY 2023
#
# This is a DEMO of the TCC's main facial recognition module by Eduardo Korte and
# Rordrigo Avelar.
# Install the packages from the requirements.txt file that can be found next to the code
# source on official github at
#
#      http://www.github.com/DUDUKorte
#
# This Demo must be run directly from the main file "main.py"
# to work correctly.
#
# THIS IS A DEMO THAT IS STILL UNDER DEVELOPMENT. PROBLEMS MAY OCCUR
# Please report issues on the project's github.

"""Facial Recognition Main Module DEMO"""


import cv2
import time
import demoFaceDetection as demoFaceDetection
from consolemenu import *
from consolemenu.items import *
from consolemenu.format import *
from colors import color


def start_mediapipe_face_detection(show_fps = False, show_camera_info = False, draw = False, upsample = False, face_encoder = False):
    """MEDIAPIPE Face Detection System, JUST DETECT FACES AND/OR DRAW IT (Boxes and/or Landmarks)
    Less accurate but VERY fast in CPUs
    
    Args:
      show_fps: Will show the current FPS of the OpenCV's window.

      show_camera_info: Will show your camera info in screen (FPS, Resolution).

      draw: Will draw landmarks and boxes in screen. DONT USE WITH UPSAMPLE TRUE.

      upsample: Make detection a LITTLE more accurate by checking 2x faces on the frame.
        DONT USE WITH DRAW TRUE. 
    """
    
    previousTime = 0 #To check camera fps

    while True:
        success, frame = CAMERA_SOURCE.read()
        print(f'----------------------------------------------------------')
        frame_time_st = time.time()
        
        if success: # If the frame was successfuly readed
            
            locations_time_st = time.time()

            #demoFDmp.draw_landmarks_face(frame) # Draw Landmarks in each face in the frame
            #demoFDmp.draw_detections_face(frame) # Draw Boxes in each face in the frame
            locations = demoFDmp.get_face_positions(frame, draw) # Get the location from each face in the frame
            
            locations_time_end = time.time()
            locations_time = locations_time_end - locations_time_st
            print(f'Locations time = {locations_time}')

            # Face Detection Section
            if locations: # If Recieve any face location coordinates
                for face_location in zip(locations): # Gets each face in locations(tuple) >> thats why need to use zip()
                    face_location = face_location[0] # BUG_FIX: face_location = (values,) have 2 positions
                    top_left = (face_location[0], face_location[3])
                    bottom_right = (face_location[2], face_location[1])

                    pixelsplus = 10
                    # Create a new frame with the recognized person's face framed
                    newFrame = frame[face_location[1]-pixelsplus if face_location[1]-pixelsplus > 0 else 0 :face_location[3]+pixelsplus if face_location[3]+pixelsplus > 0 else 0 ,
                                     face_location[0]-pixelsplus if face_location[0]-pixelsplus > 0 else 0:face_location[2]+pixelsplus if face_location[2]+pixelsplus > 0 else 0]
                    cv2.imshow('rosto enquadrado 1', newFrame)
                    cv2.waitKey(1)
                    # Rectangle in recognized person's face
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 1)
                    
                    # Face Detection Upsampled
                    # Try to re-detect a face on the new frame
                    if upsample and not draw: #BUG: If you try to use upsample AND draw landmarks at the same time, the upsample will NOT work correctly
                        if newFrame.any(): # Ensures that this new frame exists
                            locations = demoFDmp.get_face_positions(newFrame, False, True) # Get the location from each face in the frame

                            # Face Detection Section
                            if locations: # If Recieve any face location coordinates
                                for face_location in zip(locations): # Gets each face in locations(tuple) >> thats why need to use zip()
                                    face_location = face_location[0] # BUG_FIX: face_location = (values,) have 2 positions
                                    top_left = (face_location[0], face_location[3])
                                    bottom_right = (face_location[2], face_location[1])
                                    new_redetected_frame = newFrame[face_location[1] if face_location[1] > 0 else 0 :face_location[3] if face_location[3] > 0 else 0 ,
                                                                    face_location[0] if face_location[0] > 0 else 0:face_location[2] if face_location[2] > 0 else 0]
                                    
                                    # Show the new frame redetected
                                    cv2.rectangle(newFrame, top_left, bottom_right, (0, 255, 0), 1)
                                    cv2.imshow('2', new_redetected_frame)
                                    cv2.waitKey(1)

                    #TODO: Definir distância mínima para fazer o reconhecimento facial na pessoa mais próxima á câmera
                    #=======================================================
                    #TODO: Face Recognition/Encoding Goes Here
                    if face_encoder: 
                            encoding_time_st = time.time()
                            encoded_face = demoFaceDetection.get_fastest_encoded_face(frame, face_location)
                            encoding_time_end = time.time()
                            encoding_time = encoding_time_end - encoding_time_st # Calculate the time to locate all the faces in frame
                            print(f'Encoding Time = {encoding_time}')
                    #=======================================================


            # Calculate the FPS and show on the screen
            currentTime = time.time()
            fps = 1/(currentTime - previousTime)
            previousTime = currentTime
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 3) if show_fps else 0
            
            # Show the camera infos. on the screen
            demoFaceDetection.show_camera_info(CAMERA_SOURCE, frame) if show_camera_info else 0

            frame_time_end = time.time()
            frame_time = frame_time_end - frame_time_st
            print(f'Frame Time = {frame_time}')
            print(f'----------------------------------------------------------\n')

            #Update frame
            cv2.imshow('Camera', frame)
            del frame
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


def start_facerecognition_face_detection(show_fps = False, show_camera_info = False, show_stetic_landmarks = False, face_encoder = False, optimization_mode = False):
    """FACE_RECOGNITION Face Detection System, JUST DETECT FACES AND/OR DRAW IT (Boxes)
    More accurate, but slower in CPUs
    
    Args:
      show_fps: Will show the current FPS of the OpenCV's window.

      show_camera_info: Will show your camera info in screen (FPS, Resolution).

      draw: Will draw landmarks and boxes in screen. DONT USE WITH UPSAMPLE TRUE.

      upsample: Make detection a LITTLE more accurate by checking 2x faces on the frame.
        DONT USE WITH DRAW TRUE. 
    """

    # ONLY FOR DEBUGGING PURPOSES
    #====================================================================================
    optimization_mode_test = optimization_mode
    # Optimization mode Settings:
    current_fps = 1
    fps_limit = FPS/2
    #====================================================================================

    previousTime = 0 # To check camera fps

    while True:
        frame_time_st = time.time() # start counting the frame time
        success, frame = CAMERA_SOURCE.read()
        face_detected = False


        if success:# If the frame was successfuly readed
            
            #ONLY FOR DEBUGGING PURPOSES >>DEMO TEST<<
            #====================================================================================
            if optimization_mode_test and success:
                if current_fps >= fps_limit: # Verify if need to reset fps variable
                    current_fps = 0
                
                if current_fps == 0: # Verify if can do the face detection

                    locations = demoFDfr.get_face_locations(frame) # Get the face locations

                    if locations: # If found any face in the frame
                        face_detected = True
                        for face_location in zip(locations):
                            face_location = face_location[0]# BUG_FIX: face_location = (values,) have 2 positions

                            pixelsplus = 10 # Safe margin to show detected faces
                            # Crop the current frame to get just the person's face area
                            newFrame = frame[face_location[0]-pixelsplus:face_location[2]+pixelsplus,
                                            face_location[3]-pixelsplus:face_location[1]+pixelsplus] 
                            try: # Ensures that the newFrame positions are valid
                                cv2.imshow('Detected Face', newFrame)
                                cv2.waitKey(1)
                            except:
                                print("OUT OF FRAME")

                            #TODO: Definir distância mínima para fazer o reconhecimento facial na pessoa mais próxima á câmera
                            #=======================================================
                            #TODO: Face Recognition/Encoding Goes Here
                            if face_encoder: 
                                encoding_time_st = time.time()
                                #encoded_face = demoFaceDetection.get_fastest_encoded_face(frame, face_location)
                                encoded_face = demoFaceDetection.get_encoded_face(frame, face_location) # Default face encoding, use already detected face locations and (upsample)X times face encoding
                                encoding_time_end = time.time()
                                encoding_time = encoding_time_end - encoding_time_st # Calculate the time to locate all the faces in frame
                                print(f'Encoding Time = {encoding_time}')
                            #=======================================================

                current_fps+=1
            #====================================================================================
            
            else:
                
                print(f'----------------------------------------------------------')
                locations_time_st = time.time()
                locations = demoFDfr.get_face_locations(frame) # Get the locations of eah face in the frame
                locations_time_end = time.time()
                locations_time = locations_time_end - locations_time_st # Calculate the time to locate all the faces in frame
                print(f'Locations Time = {locations_time}')

                if locations:
                    face_detected = True
                    for face_location in zip(locations):
                        face_location = face_location[0]# BUG_FIX: face_location = (values,) have 2 positions

                        pixelsplus = 10 # Safe margin to show detected faces
                        newFrame = frame[face_location[0]-pixelsplus:face_location[2]+pixelsplus,
                                        face_location[3]-pixelsplus:face_location[1]+pixelsplus] # Crop the current frame to get just the person's face area
                        try: # Ensures that the newFrame positions are valid
                            cv2.imshow('Detected Face', newFrame)
                            cv2.waitKey(1)
                        except:
                            print("OUT OF FRAME")
                        
                        #TODO: Definir distância mínima para fazer o reconhecimento facial na pessoa mais próxima á câmera
                        #=======================================================
                        #TODO: Face Recognition/Encoding Goes Here
                        if face_encoder: 
                            encoding_time_st = time.time()
                            # Gets the encoded face from the location gotted before
                            #encoded_face = demoFaceDetection.get_fastest_encoded_face(frame, face_location) # Use already detected face locations and dont upsample face encoding 
                            # More encoded face functions options to use
                            #encoded_face = demoFaceDetection.get_accurative_encoded_face(frame, face_location) # Redetect Faces in image and upsample 1x times face encoding
                            encoded_face = demoFaceDetection.get_encoded_face(frame, face_location, re_samples=0) # Default face encoding, use already detected face locations and (upsample)X times face encoding
                            encoding_time_end = time.time()
                            encoding_time = encoding_time_end - encoding_time_st # Calculate the time to locate all the faces in frame
                            print(f'Encoding Time = {encoding_time}')
                        #=======================================================


            # LANDMARKS STETIC ONLY (4-5 FPS LESS)
            demoFDmp.draw_landmarks_face(frame, show_stetic_landmarks) if show_stetic_landmarks else 0


            # Calculate the FPS and show on the screen
            currentTime = time.time()
            fps = 1/(currentTime - previousTime)
            previousTime = currentTime
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 3) if show_fps else 0
            
            # Show the camera infos. on the screen
            demoFaceDetection.show_camera_info(CAMERA_SOURCE, frame) if show_camera_info else 0

            
            frame_time_end = time.time()
            frame_time = frame_time_end - frame_time_st
            print(f'Frame Time = {frame_time}')
            print(f'Face Detected = {face_detected}')
            print(f'----------------------------------------------------------\n')
            # Update frame
            cv2.imshow('Camera', frame)
            del frame
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


def start_face_recognition():
    pass

def face_validation():
    pass

def start_register():
    pass

def start_test_module():
    pass

def start_main_system():
    pass


# DEMO MENU for user
def ui_menu():
    

    # Change Menu formatting
    menu_format = MenuFormatBuilder().set_border_style_type(MenuBorderStyleType.HEAVY_OUTER_LIGHT_INNER_BORDER) \
        .set_prompt("-->") \
        .set_title_align('center') \
        .set_subtitle_align('center') \
        .set_left_margin(4) \
        .set_right_margin(4) \
        .show_header_bottom_border(True) \
        .show_prologue_top_border(False) \
        .show_prologue_bottom_border(True)

    menu = ConsoleMenu("Main Menu", "Face Recognition System {}".format(color("DEMOv1.0", fg='yellow')), formatter=menu_format, prologue_text="TCC - Facial Recognition Main Module DEMO - JULY 2023. \
                       \nThis is a DEMO of the TCC's main facial recognition module by Eduardo Korte and Rordrigo Avelar.\
                       \nInstall the packages from the requirements.txt file that can be found next to the code source on official github at [link]\
                       \n{}\
                       \n{} Please report issues on the {}".format(color("THIS IS A DEMO THAT IS STILL UNDER DEVELOPMENT.", fg='yellow', style='bold'), color("PROBLEMS MAY OCCUR.", fg='red', style='bold'), color("project's github.", fg='cyan')), clear_screen=True)

    # Create a formatter for submenu with ASCII style
    ascii_submenu = MenuFormatBuilder().set_border_style_type(MenuBorderStyleType.ASCII_BORDER)

    # MENU OPTIONS
    #======================================================================================================
    # Main System Submenu.
    mainSysMenuCs = ConsoleMenu("Main System", "Sorry, this module is Under Development. :( \nTry Again Later.",
                            formatter=ascii_submenu)
    # Menu item for add on main menu
    mainSysMenu = SubmenuItem("Start Main System", submenu=mainSysMenuCs)
    mainSysMenu.set_menu(menu)

    #======================================================================================================
    # Test Module Submenu
    testMenuCs = ConsoleMenu("Test Module", "Sorry, this module is Under Development. :( \nTry Again Later.",
                           formatter=ascii_submenu
                           )
    # Menu item for add on main menu
    testMenu = SubmenuItem("Start Test Module", submenu=testMenuCs)
    testMenu.set_menu(menu)

    #======================================================================================================
    # Face Detection MediaPipe Test
    mpFaceDetectionMenuCs = ConsoleMenu("Face Detecion {} Test".format(color("Mediapipe", fg='blue')), "{} Face Detection System, JUST DETECT FACES AND/OR DRAW IT (Boxes and/or Landmarks)\
                                      Less accurate but VERY fast in CPUs".format(color("MEDIAPIPE", fg='blue')), formatter=menu_format)
    # Menu item for add on main and sub menu
    start_detection_mp = FunctionItem("Start Face Detection", start_mediapipe_face_detection, [True, True, True])
    start_detection_upsampled_mp = FunctionItem("Start Face Detection w/Upsample", start_mediapipe_face_detection, [True, True, False, True])
    start_detection_wo_landmarks_mp = FunctionItem("Start Face Detection w/o Landmarks", start_mediapipe_face_detection, [True, True, False])
    # Adding items/Options to submenu
    mpFaceDetectionMenuCs.append_item(start_detection_mp)
    mpFaceDetectionMenuCs.append_item(start_detection_upsampled_mp)
    mpFaceDetectionMenuCs.append_item(start_detection_wo_landmarks_mp)
    # Initialize Submenu
    mpFaceDetectionMenu = SubmenuItem("Start {} Face Detecion Test".format(color("Mediapipe", fg='blue')), submenu=mpFaceDetectionMenuCs)
    mpFaceDetectionMenu.set_menu(menu)

    #======================================================================================================
    # Face Detection Face_Recognition Test
    frFaceDetectionMenuCs = ConsoleMenu("Face Detection {} Test".format(color("Face_Recognition", fg='green')), "{} Face Detection System, JUST DETECT FACES AND/OR DRAW IT (Boxes)\
                                        \nMore accurate, but slower in CPUs".format(color("FACE_RECOGNITION", fg='green')), formatter=menu_format)
    # Menu item for add on main and sub menu
    start_detection_fr = FunctionItem("Start Face Detection", start_facerecognition_face_detection, [True, True])
    start_detection_wLandmarks_fr = FunctionItem("Start Face Detection w/Landmarks {}".format(color("(4-5 FPS Less)", fg='red')), start_facerecognition_face_detection, [True, True, True])
    # Adding items/Options to submenu
    frFaceDetectionMenuCs.append_item(start_detection_fr)
    frFaceDetectionMenuCs.append_item(start_detection_wLandmarks_fr)
    # Initialize Submenu
    frFaceDetectionMenu = SubmenuItem("Start {} Face Detection Test".format(color("Face_Recognition", fg='green')), submenu=frFaceDetectionMenuCs)
    frFaceDetectionMenu.set_menu(menu)

    #======================================================================================================
    # Face Encoding Meiapipe Test
    mpFaceEncodingMenuCs = ConsoleMenu("Face Encoding {} Test".format(color("Mediapipe", fg='blue')), "{} Face Encoding System, Detect AND Encode faces on camera and draw it (boxes and/or landmarks)".format(color("MEDIAPIPE", fg='blue')),
                                       formatter=menu_format)
    # Menu item for add on main and sub menu
    start_encoding_mp = FunctionItem("Start Face Encoding", start_mediapipe_face_detection, [True, True, True, False, True])
    start_encoding__wo_landmarks_mp = FunctionItem("Start Face Encoding w/o Landmarks", start_mediapipe_face_detection, [True, True, False, False, True])
    # Adding items/Options to submenu
    mpFaceEncodingMenuCs.append_item(start_encoding_mp)
    mpFaceEncodingMenuCs.append_item(start_encoding__wo_landmarks_mp)
    # Initialize Submenu
    mpFaceEncodingMenu = SubmenuItem("Start {} Face Encoding Test".format(color("Mediapipe", fg='blue')), submenu=mpFaceEncodingMenuCs)
    mpFaceEncodingMenu.set_menu(menu)

    #======================================================================================================
    # Face Encoding Face_Recognition Test
    frFaceEncodingMenuCs = ConsoleMenu("Face Encoding {} Test".format(color("Face_Recognition", fg='green')), "{} Face Encoding System, Detect AND Encode faces on camera and draw it (boxes and/or landmarks)".format(color("FACE_RECOGNITION", fg='green')))
    # Menu item for add on main and sub menu
    start_encoding_fr = FunctionItem("Start Face Encoding", start_facerecognition_face_detection, [True, True, False, True])
    start_encoding_wLandmarks_fr = FunctionItem("Start Face Encoding w/Landmarks {}".format(color("(4-5 FPS Less)", fg='red')), start_facerecognition_face_detection, [True, True, True, True])
    start_encoding_wOptimization_fr = FunctionItem("Start Face Encoding w/{}".format(color("Otimization Mode (DEMO)", fg='green')), start_facerecognition_face_detection, [True, True, False, True, True])
    # Adding items/Options to submenu
    frFaceEncodingMenuCs.append_item(start_encoding_fr)
    frFaceEncodingMenuCs.append_item(start_encoding_wLandmarks_fr)
    frFaceEncodingMenuCs.append_item(start_encoding_wOptimization_fr)
    # Initialize Submenu
    frFaceEncodingMenu = SubmenuItem("Start {} Face Encoding Test".format(color("Face_Recognition", fg='green')), submenu=frFaceEncodingMenuCs)
    frFaceEncodingMenu.set_menu(menu)
    #======================================================================================================

    # Add all the items to the main menu
    # Start Main System
    menu.append_item(mainSysMenu)
    # Start Test Module
    menu.append_item(testMenu)
    # Start Face Detection Mediapipe Test
    menu.append_item(mpFaceDetectionMenu)
    # Start Face Detection Face_Recognition Test
    menu.append_item(frFaceDetectionMenu)
    # Start Face Encoding Mediapipe Test
    menu.append_item(mpFaceEncodingMenu)
    # Start Face Encoding Face_Recognition Test
    menu.append_item(frFaceEncodingMenu)

    # Show the menu
    menu.start()
    menu.join()
    
    # Verify if the user really wants to exit the menu
    print('    Are you sure you want to exit? (y/n)')
    print('    --> ', end='')
    user_input = menu.get_input()
    if user_input == 'n':
        ui_menu()
    elif user_input == 'hello there':
        message = demoFaceDetection.show_hello_message()
        print('{}'.format(color(message, style='negative')))
        if CAMERA_SOURCE:# If Camera source still exists
            CAMERA_SOURCE.release() # release the camera source by freeing the memory
        input()


if __name__ == '__main__':
    DEBUG = False# ONLY FOR DEBUGGING PURPOSES
    print('INFO: Initializing Camera Source...')
    CAMERA_SOURCE = cv2.VideoCapture(0)
    print('INFO: Camera initialized!\n')

    # Try to use the first camera avaible on User's PC TODO: testar se esse algoritmo para usar a câmera funciona de fato
    # for i in range(0,5):
    #     CAMERA_SOURCE = cv2.VideoCapture(i)
    #     success, frame = CAMERA_SOURCE.read()
    #     if success:
    #         break
    #     else:
    #         pass
    
    print('INFO: Getting Camera info...')
    FPS = CAMERA_SOURCE.get(5)
    print('INFO: Camera info get successfuly!\n')

    print('===========================================================')
    print('INFO: Verifying GPU Acceleration...')
    print('Face_Recognition WILL USE GPU ACCELERATION (CUDA)') if demoFaceDetection.verf_gpu_acceleration() else print('INFO: Face_Recognition WILL NOT USE GPU ACCELERATION (CUDA)')
    print('===========================================================\n')

    print('INFO: Creating Mediapipe Objects...')
    # Mediapipe Object Creation
    demoFDmp = demoFaceDetection.FaceDetectionMP(CAMERA_SOURCE, draw_landmark=True, min_face_detection_confidence=0.5, draw_circle_radius=1, max_num_faces_landmark=2)
    demoFDmp.draw_landmark_style = 1
    print('INFO: Mediapipe objects created!\n')

    print('INFO: Creating face_recognition object...')
    # FaceRecognition Object Creation
    # Models: HOG (Better for use in CPUs) or CNN (better with GPU acceleration)
    demoFDfr = demoFaceDetection.FaceDetectionFR(model='hog', draw_detections=True, locations_upsample=0, color=(255, 255, 255), thickness=1) 
    print('face_recognition object created!\n')

    print('INFO: Verifying Debug variables...\n')
    # ONLY FOR DEBUGGING PURPOSES
    if DEBUG:
        print('Starting debugging Test...')
        #start_mediapipe_face_detection(show_fps=True, show_camera_info=True, draw=True, upsample=False)
        start_facerecognition_face_detection(show_fps=True, show_camera_info=True, show_stetic_landmarks=False, face_encoder=True)

    input('Press ENTER to Start the Main System Menu...')
    ui_menu() # Start the main menu6

    if CAMERA_SOURCE: # If Camera source still exists
        CAMERA_SOURCE.release() # release the camera source by freeing the memory

