# TCC - Reconhecimento Facial Módulo Principal DEMO - JULHO 2023
# 
# Esta é uma DEMO do módulo principal de reconhecimento facial do TCC de Eduardo Korte e
# Rordrigo Avelar.
# Instale os pacotes do arquivo requirements.txt que pode ser encontrado junto ao código
# fonte no github oficial em
#
#      http://www.github.com/DUDUKorte
# 
# Esta Demo deve ser executada diretamente do arquivo principal "demoFaceRecognition.py"
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
# This Demo must be run directly from the main file "demoFaceRecognition.py"
# to work correctly.
#
# THIS IS A DEMO THAT IS STILL UNDER DEVELOPMENT. PROBLEMS MAY OCCUR
# Please report issues on the project's github.

import cv2
import mediapipe as mp
import face_recognition, dlib

def get_fastest_encoded_face(frame, locations):
    locations = [locations]
    return face_recognition.face_encodings(frame, locations, 0, "large")

def get_accurative_encoded_face(frame, locations):
    locations = [locations]
    return face_recognition.face_encodings(frame, None, 1, "large")

def get_encoded_face(frame, locations, re_samples = 1):
    locations = [locations]
    return face_recognition.face_encodings(frame, locations, re_samples, "large")

def show_hello_message():
    hello_message = " __    __   _______  __       __        ______      .___________. __    __   _______ .______       _______ \n|  |  |  | |   ____||  |     |  |      /  __  \     |           ||  |  |  | |   ____||   _  \     |   ____|\n|  |__|  | |  |__   |  |     |  |     |  |  |  |    `---|  |----`|  |__|  | |  |__   |  |_)  |    |  |__   \n|   __   | |   __|  |  |     |  |     |  |  |  |        |  |     |   __   | |   __|  |      /     |   __|  \n|  |  |  | |  |____ |  `----.|  `----.|  `--'  |        |  |     |  |  |  | |  |____ |  |\  \----.|  |____ \n|__|  |__| |_______||_______||_______| \______/         |__|     |__|  |__| |_______|| _| `._____||_______|"
    return hello_message

def verf_gpu_acceleration():
    return dlib.DLIB_USE_CUDA

def show_camera_info(camera_source: cv2.VideoCapture, frame, fps = None, width = None, height = None):

    """Draw the default camera Info. (fps and resolution).

    Args:
      camera_source: The camera to get the info

      frame: The frame to draw the info (optional)

      fps: Specify the framerate on the camera (optional)

      width: Specify the Camera Widht (optional)

      height: Specify the Camera Height (optional)
    """
    fps = camera_source.get(5)
    width = camera_source.get(3)
    height = camera_source.get(4)
    cv2.putText(frame, f'FPS: {int(fps)} RESOLUTION: {int(width)}x{int(height)}', (10, int(height)-2), cv2.FONT_HERSHEY_PLAIN, 2, (100,255,100), 1)

#================================================================================
class FaceDetectionMP:
    camera_source_global = None
    WIDTH  = None
    HEIGHT = None
    FPS = None

    def __init__(self,
                 camera_source: cv2.VideoCapture,
                 min_face_detection_confidence = 0.5,
                 model_selection = 0,
                 draw_detections = True,
                 draw_landmark = False,
                 max_num_faces_landmark = 2,
                 draw_thickness = 1,
                 draw_circle_radius = 1,
                 ):
        """Initializes a FaceDetectionMediaPipe Object to use Mediapipe Face Detection algorithm.
        
        Args:
          min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
          detection to be considered successful.

          model_selection: 0 or 1. 0 to select a short-range model that works
            best for faces within 2 meters from the camera, and 1 for a full-range
            model best for faces within 5 meters.

          draw_detections: Define if the face detections will be drawed or not (Rectangle).
          
          draw_landmarks: Define if the face landmarks will be drawed or not.

          max_num_faces_landmark: Maximum number of faces to detect in landmark.

          draw_thickness: Thickness to use in landmarks and face detection draw.

          draw_circle_radius: Circle radius to use in landmarks draw
        """
        global camera_source_global, WIDTH, HEIGHT, FPS
        camera_source_global = camera_source
        del camera_source
        WIDTH  = camera_source_global.get(3)  # float fixed camera width
        HEIGHT = camera_source_global.get(4)  # float fixed camera height
        FPS = camera_source_global.get(5) #float fixed fps

        #Self.declarations
        self.draw_landmark = draw_landmark
        self.draw_detections = draw_detections

        #FaceMesh objects to create landmarks if draw_landmarks is True
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=max_num_faces_landmark,
                                                 refine_landmarks=True,
                                                 min_detection_confidence=0.9,
                                                 min_tracking_confidence=0.9
                                                 )

        #FaceDetection objects to create detections
        mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=min_face_detection_confidence, model_selection = model_selection)
        del mpFaceDetection

        #Drawing Objects to draw landmarks and detections
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = draw_thickness , circle_radius = draw_circle_radius)
        
        #Defining the Landmark Style to draw
        """Landmarks styles:
            0 - FACEMESH_CONTOURS (contours of all) 1 - FACEMESH_TESSELATION(All points connected to each other)
            2 - FACEMESH_LEFT_EYE (Olho esquerdo)   3 - FACEMESH_LEFT_EYEBROW(sobrancelha do olho esquerdo) 
            4 - FACEMESH_RIGHT_EYE(Olho direito)    5 - FACEMESH_RIGHT_EYEBROW(sobrancelha do olho direito)
            6 - FACEMESH_FACE_OVAL(Contorno rosto)  7 - FACEMESH_LIPS (Lábios)
        """
        lm_styles = [self.mpFaceMesh.FACEMESH_CONTOURS,
                     self.mpFaceMesh.FACEMESH_TESSELATION,
                     self.mpFaceMesh.FACEMESH_LEFT_EYE,
                     self.mpFaceMesh.FACEMESH_LEFT_EYEBROW,
                     self.mpFaceMesh.FACEMESH_RIGHT_EYE,
                     self.mpFaceMesh.FACEMESH_RIGHT_EYEBROW,
                     self.mpFaceMesh.FACEMESH_FACE_OVAL,
                     self.mpFaceMesh.FACEMESH_LIPS
                  ]
        self.lm_styles = lm_styles
        del lm_styles
        del draw_landmark
        if self.draw_landmark: 
            self.draw_landmark_style = 3
            self._draw_landmark_style = self.lm_styles[self.draw_landmark_style]

    #Landmark Style Restrictions
    def onlyIFLandmark(self):
        if self.draw_landmark:
            @property
            def draw_landmark_style(self):
                return self.temp_draw_landmark_style
            @draw_landmark_style.setter
            def draw_landmark_style(self, value):
                if isinstance(value, int) and value >= 0 and value <= 7:
                    self.temp_draw_landmark_style = value
                else:
                    raise ValueError("The value must be an integer between 0-7")
                

    #Functions
    def detect_faces(self, frame, landmark_optimized = False):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks_results = None
        if self.draw_landmark: 
            landmarks_results = self.faceMesh.process(frameRGB)
        if not landmark_optimized:
            detection_results = self.faceDetection.process(frameRGB)
            return [detection_results, landmarks_results]
        else:
            return ['landmark optimized as True', landmarks_results]

    def get_face_positions(self, frame, draws = False, upsampled_frame = False):
        """Detect faces on image and return a list of tuples with the position of each one.
        Args:
          frame: the frame/image to detect faces
          draws: will draw the faces detectios and/or face landmarks in image (based on draw_detections and draw_locations variables)
        """
        global HEIGHT, WIDTH

        detection_results, landmarks_results = self.detect_faces(frame) #Get detections result from the face detection function

        #Drawing Warning
        if not (self.draw_landmark or self.draw_detections) and draws:
            print('WARNING: you are trying to draw with the drawing variables as False')

        if landmarks_results.multi_face_landmarks if self.draw_landmark and draws else 0: #If detects any face in the image
                self._draw_landmark_style = self.lm_styles[self.draw_landmark_style]
                for faceLandmarks in landmarks_results.multi_face_landmarks: #Will use the value of landmarks for each face detected
                    self.mpDraw.draw_landmarks(frame, faceLandmarks, self._draw_landmark_style, self.drawSpec, self.drawSpec) #draw the landmark on image

        #Face Detection
        if detection_results.detections: #If detects any face in the image
            detection_locations = []
            for detections in detection_results.detections: #Will use the values of each face detected
                if draws:
                    self.mpDraw.draw_detection(frame, detections, self.drawSpec, self.drawSpec) if self.draw_detections else 0

                relative_bounding_box = detections.location_data.relative_bounding_box

                
                HEIGHT, WIDTH, channels = frame.shape
                del channels

                initial_x_y = self.mpDraw._normalized_to_pixel_coordinates(relative_bounding_box.xmin, relative_bounding_box.ymin, int(WIDTH), int(HEIGHT)-30) #Get the initial x and y coordinates of the face
                finals_x_y = self.mpDraw._normalized_to_pixel_coordinates(relative_bounding_box.xmin + relative_bounding_box.width, relative_bounding_box.ymin + relative_bounding_box.height, int(WIDTH), int(HEIGHT)) #Get the final x and y coordinates of the face
                try:
                    detection_location = (initial_x_y[0], initial_x_y[1], finals_x_y[0], finals_x_y[1]) #Formatted data to use later
                    detection_locations.append(detection_location)
                except:
                    print('OUT OF FRAME')
            return detection_locations
            
    def draw_detections_face(self, frame):
        """Detect faces on image and draw a rectangle in each one.
        Args:
          frame: the frame/image to detect faces and draw the rectangles
        """
        detection_results = self.detect_faces(frame)[0]
        if detection_results.detections:
            for detections in detection_results.detections:
                self.mpDraw.draw_detection(frame, detections, self.drawSpec, self.drawSpec) if self.draw_detections else 0

    def draw_landmarks_face(self, frame, optimized = False):
        """Detect faces on image and draw the landmarks on each one (up to the defined maximum).
        Args:
          frame: the frame/image to detect faces and draw the landmarks

          optimized: use if you just want to draw the landmarks and dont
            detect the faces
        """
        landmarks_results = self.detect_faces(frame, optimized)[1] #Get the landmarks results from the face detection function
        if not self.draw_landmark:
            print('WARNING: you are trying to draw landmarks with the "draw_landmark" as False')
        if landmarks_results.multi_face_landmarks if self.draw_landmark else 0: #If detects any face in the image
                self._draw_landmark_style = self.lm_styles[self.draw_landmark_style]
                for faceLandmarks in landmarks_results.multi_face_landmarks: #Will use the value of landmarks for each face detected
                    self.mpDraw.draw_landmarks(frame, faceLandmarks, self._draw_landmark_style, self.drawSpec, self.drawSpec) #draw the landmark on image


#================================================================================
class FaceDetectionFR:

    def __init__(self, model = 'hog',
                 locations_upsample = 0,
                 draw_detections = False,
                 thickness = 1,
                 color = (255,0,0)
                 ):
        """Initializes a FaceDetection_FaceRecognition Object to use face_recognition Face Detection algorithm.
        
        Args:
          model: Which face detection model to use. “hog” is less accurate but
            faster on CPUs. “cnn” is a more accurate deep-learning model which
            is GPU/CUDA accelerated (if available). The default is “hog”.

          locations_upsample: How many times to upsample the image looking for faces.
            Higher numbers find smaller faces.
          
          draw_detections: If you want to draw boxes around the detected faces
            in the image.
            
          thickness: Thickness to draw a box around the faces.
          
          color: Color to draw the boxes on the frame.
        """
        #Self declarations
        self.model = model
        self.locations_upsample = locations_upsample
        self.draw_detections = draw_detections
        self.thickness = thickness
        self.color = color

    #Functions
    def get_face_locations(self, frame):
        face_locations = face_recognition.face_locations(frame, self.locations_upsample, self.model)
        if face_locations:
            if self.draw_detections:
                for face_location in zip(face_locations):
                    face_location = face_location[0]
                    top_left = (face_location[3], face_location[0])
                    bottom_right = (face_location[1], face_location[2])
                    cv2.rectangle(frame, top_left, bottom_right, self.color, self.thickness)
            return face_locations

#================================================================================
