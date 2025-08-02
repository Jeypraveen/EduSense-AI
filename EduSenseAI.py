import tkinter as tk
from tkinter import messagebox, ttk, filedialog, scrolledtext
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import threading
from collections import defaultdict, deque, Counter
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F
import csv
import mediapipe as mp
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

class EduSenseAI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EduSense AI - Comprehensive Educational AI Platform")
        self.root.geometry("1400x900")
        self.root.configure(background='#2c3e50')
        
        # Initialize shared components
        self.setup_shared_models()
        
        # Initialize module-specific variables
        self.init_attendance_vars()
        self.init_emotion_vars()
        self.init_proctoring_vars()
        
        # Setup main GUI
        self.setup_main_gui()
        
    def setup_shared_models(self):
        """Setup models shared across modules"""
        try:
            # FaceNet models (shared)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.mtcnn = MTCNN(
                keep_all=True, 
                device=self.device,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                post_process=True
            )
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            # MediaPipe Face Mesh (shared)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # OpenCV cascade for fallback
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to initialize shared models: {e}")
    
    def init_attendance_vars(self):
        """Initialize attendance system variables"""
        self.attendance_active = False
        self.attendance_cap = None
        self.setup_directories()
        
    def init_emotion_vars(self):
        """Initialize emotion detection variables"""
        self.emotion_active = False
        self.emotion_cap = None
        self.HISTORY_LENGTH = 7
        self.emotion_history = deque(maxlen=self.HISTORY_LENGTH)
        self.cognitive_state_history = deque(maxlen=self.HISTORY_LENGTH)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.cognitive_labels = ['Attentive', 'Distracted', 'Drowsy']
        self.cognitive_states_report = []
        self.emotions_report = []
        self.session_start_time = None
        self.total_frames_processed = 0
        
        # Try to load emotion model
        try:
            model_path = r'C:\Projects\EduSense AI\FERTest3.keras'
            if os.path.exists(model_path):
                self.emotion_model = load_model(model_path)
            else:
                self.emotion_model = None
                print("Emotion model not found at specified path")
        except Exception as e:
            self.emotion_model = None
            print(f"Failed to load emotion model: {e}")
    
    def init_proctoring_vars(self):
        """Initialize exam proctoring variables"""
        self.proctoring_active = False
        self.screening_active = False
        self.proctoring_cap = None
        self.screening_cap = None
        self.known_embeddings = None
        self.name_mapping = None
        self.student_name = "Unknown"
        self.student_id = None
        self.violation_log = []
        self.cognitive_states = {'attentive': 0, 'distracted': 0, 'drowsy': 0, 'absent': 0}
        self.proctoring_total_frames = 0
        self.proctoring_session_start_time = None
        self.proctoring_cognitive_history = deque(maxlen=10)
        
        # Try to load YOLO model
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            self.banned_objects = {
                67: 'cell phone', 73: 'laptop', 74: 'mouse', 75: 'remote',
                76: 'keyboard', 84: 'book'
            }
        except Exception as e:
            self.yolo_model = None
            print(f"Failed to load YOLO model: {e}")
        
        # Load face embeddings
        self.load_face_data()
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['TrainingImages', 'ImagesUnknown', 'QualityCheck', 'models', 'embeddings', 'analysis_reports']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def load_face_data(self):
        """Load face embeddings for recognition"""
        try:
            embeddings_path = "embeddings/face_embeddings.npy"
            names_path = "embeddings/name_mapping.npy"
            
            if not os.path.exists(embeddings_path) or not os.path.exists(names_path):
                embeddings_path = "face_embeddings.npy"
                names_path = "name_mapping.npy"
            
            if os.path.exists(embeddings_path) and os.path.exists(names_path):
                self.known_embeddings = np.load(embeddings_path, allow_pickle=True).item()
                self.name_mapping = np.load(names_path, allow_pickle=True).item()
                print(f"Face recognition data loaded: {len(self.known_embeddings)} people")
            else:
                print("Face recognition files not found")
                
        except Exception as e:
            print(f"Error loading face data: {e}")
            self.known_embeddings = None
            self.name_mapping = None
    
    def setup_main_gui(self):
        """Setup the main GUI interface"""
        # Main container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_container, 
                              text="EduSense AI - Comprehensive Educational AI Platform",
                              bg='#2c3e50', fg='#ecf0f1', 
                              font=('Arial', 20, 'bold'))
        title_label.pack(pady=10)
        
        # Device info
        device_label = tk.Label(main_container, 
                               text=f"Computing Device: {self.device}",
                               bg='#2c3e50', fg='#f39c12', 
                               font=('Arial', 10))
        device_label.pack()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True, pady=10)
        
        # Setup individual modules
        self.setup_attendance_tab()
        self.setup_emotion_tab()
        self.setup_proctoring_tab()
        
        # Status bar
        self.status_bar = tk.Label(main_container, 
                                  text="Ready - Select a module to begin",
                                  relief=tk.SUNKEN, 
                                  anchor=tk.W,
                                  bg='#34495e', fg='#ecf0f1')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_attendance_tab(self):
        """Setup the attendance tracking tab"""
        attendance_frame = ttk.Frame(self.notebook)
        self.notebook.add(attendance_frame, text="📊 Attendance System")
        
        # Title
        title = tk.Label(attendance_frame, 
                        text="FaceNet Enhanced Face Recognition Attendance System",
                        font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Input frame
        input_frame = tk.LabelFrame(attendance_frame, text="Student Information", 
                                   font=('Arial', 12, 'bold'), padx=10, pady=10)
        input_frame.pack(fill='x', padx=20, pady=10)
        
        # Name input
        tk.Label(input_frame, text="Full Name:", font=('Arial', 11)).grid(row=0, column=0, sticky='w', pady=5)
        self.std_name = tk.Entry(input_frame, font=('Arial', 11), width=30)
        self.std_name.grid(row=0, column=1, padx=10, pady=5)
        tk.Button(input_frame, text="Clear", command=lambda: self.std_name.delete(0, 'end'),
                 bg='#e74c3c', fg='white').grid(row=0, column=2, padx=5)
        
        # ID input
        tk.Label(input_frame, text="Student ID:", font=('Arial', 11)).grid(row=1, column=0, sticky='w', pady=5)
        self.std_number = tk.Entry(input_frame, font=('Arial', 11), width=30)
        self.std_number.grid(row=1, column=1, padx=10, pady=5)
        tk.Button(input_frame, text="Clear", command=lambda: self.std_number.delete(0, 'end'),
                 bg='#e74c3c', fg='white').grid(row=1, column=2, padx=5)
        
        # Progress
        self.attendance_progress_var = tk.StringVar()
        self.attendance_progress_var.set("Ready")
        progress_label = tk.Label(attendance_frame, textvariable=self.attendance_progress_var,
                                 font=('Arial', 12, 'bold'))
        progress_label.pack(pady=5)
        
        self.attendance_progress = ttk.Progressbar(attendance_frame, length=400, mode='determinate')
        self.attendance_progress.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(attendance_frame)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="CAPTURE IMAGES\n(FaceNet Detection)", 
                 command=self.start_capture_thread,
                 bg='#27ae60', fg='white', font=('Arial', 11, 'bold'), 
                 width=20, height=3).grid(row=0, column=0, padx=10)
        
        tk.Button(button_frame, text="TRAIN MODEL\n(FaceNet Embeddings)", 
                 command=self.start_train_thread,
                 bg='#3498db', fg='white', font=('Arial', 11, 'bold'), 
                 width=20, height=3).grid(row=0, column=1, padx=10)
        
        tk.Button(button_frame, text="START TRACKING\n(Real-time)", 
                 command=self.start_attendance_thread,
                 bg='#f39c12', fg='white', font=('Arial', 11, 'bold'), 
                 width=20, height=3).grid(row=0, column=2, padx=10)
        
        tk.Button(button_frame, text="VIEW ATTENDANCE", 
                 command=self.view_attendance,
                 bg='#9b59b6', fg='white', font=('Arial', 11, 'bold'), 
                 width=20, height=3).grid(row=0, column=3, padx=10)
        
        # Notification area
        tk.Label(attendance_frame, text="Status & Notifications", 
                font=('Arial', 12, 'bold')).pack(pady=(20, 5))
        self.attendance_notification_text = tk.Text(attendance_frame, height=8, width=100, 
                                                   font=('Arial', 10))
        self.attendance_notification_text.pack(pady=5)
    
    def setup_emotion_tab(self):
        """Setup the emotion detection tab"""
        emotion_frame = ttk.Frame(self.notebook)
        self.notebook.add(emotion_frame, text="😊 Emotion Analysis")
        
        # Title
        title = tk.Label(emotion_frame, 
                        text="Facial Emotion & Cognitive State Detection",
                        font=('Arial', 16, 'bold'))
        title.pack(pady=15)
        
        # Control buttons
        button_frame = tk.Frame(emotion_frame)
        button_frame.pack(pady=10)
        
        self.emotion_webcam_btn = tk.Button(button_frame, text="Start Webcam Analysis",
                                           font=('Arial', 12), bg='#4CAF50', fg='white',
                                           command=self.start_emotion_webcam, 
                                           width=25, height=2)
        self.emotion_webcam_btn.pack(side=tk.LEFT, padx=5)
        
        self.emotion_video_btn = tk.Button(button_frame, text="Select Video for Analysis",
                                          font=('Arial', 12), bg='#4CAF50', fg='white',
                                          command=self.select_emotion_video, 
                                          width=25, height=2)
        self.emotion_video_btn.pack(side=tk.LEFT, padx=5)
        
        self.emotion_folder_btn = tk.Button(button_frame, text="Select Folder for Analysis",
                                           font=('Arial', 12), bg='#4CAF50', fg='white',
                                           command=self.select_emotion_folder, 
                                           width=25, height=2)
        self.emotion_folder_btn.pack(side=tk.LEFT, padx=5)
        
        # Instructions
        instructions = tk.Label(emotion_frame,
                               text="Instructions:\n• Choose an analysis mode above.\n• In video/webcam window, press 'Q' to quit current analysis.\n• For folder analysis, reports are saved automatically.",
                               font=('Arial', 11), bg='#e6f3ff', fg='#333',
                               relief='solid', bd=1, padx=10, pady=10)
        instructions.pack(pady=15, fill=tk.X, padx=20)
        
        # Report area
        report_frame = tk.Frame(emotion_frame)
        report_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        tk.Label(report_frame, text="Analysis Report", 
                font=('Arial', 14, 'bold')).pack()
        
        self.emotion_report_text = scrolledtext.ScrolledText(report_frame, height=15, width=90,
                                                            font=('Consolas', 10), 
                                                            bg='#ffffff', fg='#333333', 
                                                            wrap=tk.WORD)
        self.emotion_report_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.emotion_clear_btn = tk.Button(report_frame, text="Clear Report Display",
                                          font=('Arial', 12), bg='#ff6b6b', fg='white',
                                          command=self.clear_emotion_report, width=20)
        self.emotion_clear_btn.pack(pady=5)
    
    def setup_proctoring_tab(self):
        """Setup the exam proctoring tab"""
        proctoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(proctoring_frame, text="🎓 Exam Proctoring")
        
        # Main container
        main_container = tk.Frame(proctoring_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title = tk.Label(main_container, 
                        text="AI-Powered Exam Proctoring System",
                        font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Video frame
        
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)

        video_frame = tk.LabelFrame(
            main_container,
            text="Live Monitoring",
            font=('Arial', 12, 'bold')
        )
        video_frame.grid(
            row=1, column=0, columnspan=2,
            padx=5, pady=5,
            sticky='nsew'
            
        )
        

        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        self.proctoring_video_label = tk.Label(
            video_frame,
            text="Video feed will appear here",
            bg='black', fg='white'
        )
        self.proctoring_video_label.grid(
            row=0, column=0,
            sticky='nsew',
            padx=5, pady=5
        )
        

        
        # Status frame
        status_frame = tk.LabelFrame(main_container, text="System Status", 
                                    font=('Arial', 12, 'bold'))
        status_frame.grid(row=1, column=2, padx=5, pady=5, sticky='new')
        
        self.proctoring_student_label = tk.Label(status_frame, text="Student: Not Identified",
                                                font=('Arial', 11))
        self.proctoring_student_label.grid(row=0, column=0, sticky='w', pady=2)
        
        self.proctoring_status_label = tk.Label(status_frame, text="Status: Ready",
                                               font=('Arial', 11))
        self.proctoring_status_label.grid(row=1, column=0, sticky='w', pady=2)
        
        self.proctoring_cognitive_label = tk.Label(status_frame, text="State: Monitoring Off",
                                                  font=('Arial', 11))
        self.proctoring_cognitive_label.grid(row=2, column=0, sticky='w', pady=2)
        
        self.proctoring_violations_label = tk.Label(status_frame, text="Violations: 0",
                                                   font=('Arial', 11))
        self.proctoring_violations_label.grid(row=3, column=0, sticky='w', pady=2)
        
        # Control buttons
        control_frame = tk.Frame(main_container)
        control_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.proctoring_start_btn = tk.Button(control_frame, text="Start Monitoring",
                                             command=self.start_proctoring_screening,
                                             bg='#4CAF50', fg='white', 
                                             font=('Arial', 12), width=15)
        self.proctoring_start_btn.grid(row=0, column=0, padx=5)
        
        self.proctoring_stop_btn = tk.Button(control_frame, text="Stop Monitoring",
                                            command=self.stop_proctoring_monitoring,
                                            bg='#f44336', fg='white', 
                                            font=('Arial', 12), width=15,
                                            state='disabled')
        self.proctoring_stop_btn.grid(row=0, column=1, padx=5)
        
        self.proctoring_report_btn = tk.Button(control_frame, text="Generate Report",
                                              command=self.generate_proctoring_report,
                                              bg='#2196F3', fg='white', 
                                              font=('Arial', 12), width=15)
        self.proctoring_report_btn.grid(row=0, column=2, padx=5)
        
        # Log display
        log_frame = tk.LabelFrame(main_container, text="Violation Log", 
                                 font=('Arial', 12, 'bold'))
        log_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky='ew')
        
        self.proctoring_log_text = tk.Text(log_frame, height=8, width=100, 
                                          font=('Arial', 10))
        scrollbar = tk.Scrollbar(log_frame, orient='vertical', 
                                command=self.proctoring_log_text.yview)
        self.proctoring_log_text.configure(yscrollcommand=scrollbar.set)
        
        self.proctoring_log_text.grid(row=0, column=0, sticky='ew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Configure grid weights
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(3, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
    
    # ==================== ATTENDANCE SYSTEM METHODS ====================
    
    def log_attendance_message(self, message):
        """Add message to attendance notification area"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        self.attendance_notification_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.attendance_notification_text.see(tk.END)
        self.root.update()
    
    def start_capture_thread(self):
        threading.Thread(target=self.capture_images, daemon=True).start()
    
    def start_train_thread(self):
        threading.Thread(target=self.train_model, daemon=True).start()
    
    def start_attendance_thread(self):
        threading.Thread(target=self.track_attendance, daemon=True).start()
    
    def capture_images(self):
        """Enhanced image capture with MTCNN detection"""
        name = self.std_name.get().strip()
        student_id = self.std_number.get().strip()
        
        if not name or not student_id:
            messagebox.showerror("Error", "Please enter both name and student ID")
            return
            
        if not name.replace(" ", "").isalpha():
            messagebox.showerror("Error", "Name should contain only letters")
            return
            
        self.log_attendance_message(f"Starting FaceNet image capture for {name} (ID: {student_id})")
        
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Cannot access camera")
            return
            
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        
        sample_num = 0
        target_samples = 100
        quality_samples = 0
        total_frames = 0
        faces_detected = 0
        
        self.attendance_progress['maximum'] = target_samples
        self.attendance_progress['value'] = 0
        
        while sample_num < target_samples:
            ret, frame = cam.read()
            if not ret:
                self.log_attendance_message("Failed to capture frame")
                break
                
            total_frames += 1
            faces, confidences = self.detect_faces_mtcnn(frame)
            
            for i, ((x, y, w, h), confidence) in enumerate(zip(faces, confidences)):
                faces_detected += 1
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                y1 = max(0, y)
                y2 = min(frame.shape[0], y + h)
                x1 = max(0, x)
                x2 = min(frame.shape[1], x + w)
                face_region = frame[y1:y2, x1:x2]
                
                if confidence > 0.95 and face_region.size > 0 and face_region.shape[0] > 10 and face_region.shape[1] > 10:
                    sample_num += 1
                    quality_samples += 1
                    
                    try:
                        face_resized = cv2.resize(face_region, (160, 160))
                        filename = f"TrainingImages/{name}.{student_id}.{sample_num}.jpg"
                        cv2.imwrite(filename, face_resized)
                    except cv2.error as e:
                        self.log_attendance_message(f"Error saving image {sample_num}: {e}")
                        sample_num -= 1
                        continue
                    
                    self.attendance_progress['value'] = sample_num
                    self.attendance_progress_var.set(f"Captured: {sample_num}/{target_samples} - Conf: {confidence:.3f}")
                    
                    cv2.putText(frame, f"Captured: {sample_num}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Quality: {confidence:.2f}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
            cv2.putText(frame, f"Press 'q' to quit early", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('FaceNet Enhanced Face Capture', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cam.release()
        cv2.destroyAllWindows()
        
        if sample_num > 0:
            row = [student_id, name, sample_num, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            
            csv_file = 'student_details.csv'
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['ID', 'Name', 'Images_Captured', 'Date_Registered'])
                writer.writerow(row)
                
            detection_rate = (faces_detected / max(total_frames, 1)) * 100
            self.log_attendance_message(f"✓ Capture completed: {sample_num} images saved")
            self.log_attendance_message(f"Quality samples: {quality_samples}, Detection rate: {detection_rate:.1f}%")
            self.attendance_progress_var.set("Capture completed successfully!")
        else:
            self.log_attendance_message("✗ No valid images captured")
            self.attendance_progress_var.set("Capture failed - try again")
    
    def detect_faces_mtcnn(self, frame):
        """Detect faces using MTCNN"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            boxes, probs = self.mtcnn.detect(pil_image)
            
            if boxes is not None:
                faces = []
                confidences = []
                for box, prob in zip(boxes, probs):
                    if prob > 0.8:
                        x1, y1, x2, y2 = box.astype(int)
                        w = x2 - x1
                        h = y2 - y1
                        faces.append((x1, y1, w, h))
                        confidences.append(prob)
                return faces, confidences
            else:
                return [], []
                
        except Exception as e:
            self.log_attendance_message(f"MTCNN detection error: {e}")
            return [], []
    
    def extract_face_embedding(self, frame, box):
        """Extract face embedding using FaceNet"""
        try:
            if frame is None or len(box) != 4:
                return None
                
            x, y, w, h = box
            
            frame_height, frame_width = frame.shape[:2]
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = max(1, min(w, frame_width - x))
            h = max(1, min(h, frame_height - y))
            
            face_region = frame[y:y+h, x:x+w]
            if face_region.size == 0 or face_region.shape[0] < 10 or face_region.shape[1] < 10:
                return None
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            bbox = [x, y, x+w, y+h]
            
            face_tensor = self.mtcnn.extract(pil_image, [bbox], save_path=None)
            
            if face_tensor is not None and len(face_tensor) > 0:
                face_tensor = face_tensor[0]
                
                if face_tensor is None or torch.isnan(face_tensor).any():
                    return None
                    
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.resnet(face_tensor)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    
                return embedding.cpu().numpy().flatten()
            else:
                return None
                
        except Exception as e:
            self.log_attendance_message(f"Embedding extraction error: {e}")
            return None
    
    def train_model(self):
        """Train model using FaceNet embeddings"""
        self.log_attendance_message("Starting FaceNet model training...")
        
        training_path = "TrainingImages"
        if not os.path.exists(training_path) or not os.listdir(training_path):
            messagebox.showerror("Error", "No training images found. Please capture images first.")
            return
            
        image_files = [f for f in os.listdir(training_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            messagebox.showerror("Error", "No valid training images found")
            return
            
        self.log_attendance_message(f"Processing {len(image_files)} training images...")
        
        self.attendance_progress['maximum'] = len(image_files)
        self.attendance_progress['value'] = 0
        
        embeddings_db = {}
        name_mapping = {}
        processed_count = 0
        
        for i, filename in enumerate(image_files):
            try:
                parts = filename.split('.')
                if len(parts) >= 3:
                    name = parts[0]
                    student_id = int(parts[1])
                    
                    image_path = os.path.join(training_path, filename)
                    frame = cv2.imread(image_path)
                    
                    if frame is not None:
                        h, w = frame.shape[:2]
                        box = (0, 0, w, h)
                        
                        embedding = self.extract_face_embedding(frame, box)
                        
                        if embedding is not None:
                            if student_id not in embeddings_db:
                                embeddings_db[student_id] = []
                                name_mapping[student_id] = name
                                
                            embeddings_db[student_id].append(embedding)
                            processed_count += 1
                            
                self.attendance_progress['value'] = i + 1
                self.attendance_progress_var.set(f"Processing: {i+1}/{len(image_files)}")
                self.root.update()
                
            except Exception as e:
                self.log_attendance_message(f"Error processing {filename}: {e}")
                continue
                
        if not embeddings_db:
            messagebox.showerror("Error", "No valid embeddings generated")
            return
            
        self.attendance_progress_var.set("Calculating average embeddings...")
        average_embeddings = {}
        
        for student_id, embeddings_list in embeddings_db.items():
            if embeddings_list:
                embeddings_array = np.array(embeddings_list)
                average_embedding = np.mean(embeddings_array, axis=0)
                average_embedding = average_embedding / np.linalg.norm(average_embedding)
                average_embeddings[student_id] = average_embedding
                
        try:
            np.save("embeddings/face_embeddings.npy", average_embeddings)
            np.save("embeddings/name_mapping.npy", name_mapping)
            
            self.attendance_progress['value'] = len(image_files)
            self.log_attendance_message("✓ FaceNet model training completed successfully")
            self.log_attendance_message(f"✓ Generated embeddings for {len(average_embeddings)} people")
            self.log_attendance_message(f"✓ Processed {processed_count}/{len(image_files)} images")
            
        except Exception as e:
            self.log_attendance_message(f"✗ Failed to save model: {str(e)}")
            return
                
        self.attendance_progress_var.set("Training completed!")
        messagebox.showinfo("Success", f"Training completed! Generated embeddings for {len(average_embeddings)} people.")
        
        # Reload face data for other modules
        self.load_face_data()
    
    def recognize_face_facenet(self, frame, box):
        """Recognize face using FaceNet embeddings"""
        try:
            embeddings_path = "embeddings/face_embeddings.npy"
            names_path = "embeddings/name_mapping.npy"
            
            if not os.path.exists(embeddings_path) or not os.path.exists(names_path):
                return None, 0
                
            stored_embeddings = np.load(embeddings_path, allow_pickle=True).item()
            name_mapping = np.load(names_path, allow_pickle=True).item()
            
            current_embedding = self.extract_face_embedding(frame, box)
            
            if current_embedding is None:
                return None, 0
                
            best_match_id = None
            best_similarity = 0
            
            for student_id, stored_embedding in stored_embeddings.items():
                similarity = np.dot(current_embedding, stored_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = student_id
                    
            confidence = best_similarity * 100
            
            return best_match_id, confidence
            
        except Exception as e:
            self.log_attendance_message(f"Recognition error: {str(e)}")
            return None, 0
    
    def track_attendance(self):
        """Enhanced real-time attendance tracking using FaceNet"""
        embeddings_path = "embeddings/face_embeddings.npy"
        names_path = "embeddings/name_mapping.npy"
        
        if not os.path.exists(embeddings_path) or not os.path.exists(names_path):
            messagebox.showerror("Error", "No trained model found. Please train the model first.")
            return
        
        try:
            stored_embeddings = np.load(embeddings_path, allow_pickle=True).item()
            name_mapping = np.load(names_path, allow_pickle=True).item()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            return
        
        self.log_attendance_message("Starting FaceNet attendance tracking...")
        self.log_attendance_message(f"Loaded embeddings for {len(stored_embeddings)} people")
        
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Cannot access camera")
            return
            
        attendance_today = defaultdict(int)
        last_recognition = {}
        recognition_cooldown = 5
        
        attendance_file = 'attendance.csv'
        if not os.path.exists(attendance_file):
            with open(attendance_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['ID', 'Name', 'Date', 'Time', 'Status'])
                
        self.attendance_active = True
        
        while self.attendance_active:
            ret, frame = cam.read()
            if not ret:
                break
                
            faces, confidences = self.detect_faces_mtcnn(frame)
            current_time = time.time()
            
            for (x, y, w, h), detection_conf in zip(faces, confidences):
                if detection_conf > 0.9:
                    student_id, recognition_conf = self.recognize_face_facenet(frame, (x, y, w, h))
                    
                    if student_id is not None and recognition_conf > 85:
                        name = name_mapping.get(student_id, "Unknown")
                        
                        if (student_id not in last_recognition or 
                            current_time - last_recognition[student_id] > recognition_cooldown):
                            
                            timestamp = datetime.datetime.now()
                            date_str = timestamp.strftime('%Y-%m-%d')
                            time_str = timestamp.strftime('%H:%M:%S')
                            
                            with open(attendance_file, 'a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([student_id, name, date_str, time_str, 'Present'])
                                
                            attendance_today[student_id] += 1
                            last_recognition[student_id] = current_time
                            
                            self.log_attendance_message(f"✓ Attendance: {name} (ID: {student_id}, Conf: {recognition_conf:.1f}%)")
                            
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{name} ({recognition_conf:.1f}%)"
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        conf_text = f"{recognition_conf:.1f}%" if student_id is not None else "Unknown"
                        cv2.putText(frame, f"Unknown ({conf_text})", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if recognition_conf < 70:
                            y1 = max(0, y)
                            y2 = min(frame.shape[0], y + h)
                            x1 = max(0, x)
                            x2 = min(frame.shape[1], x + w)
                            
                            face_region = frame[y1:y2, x1:x2]
                            
                            if face_region.size > 0 and face_region.shape[0] > 10 and face_region.shape[1] > 10:
                                unknown_count = len([f for f in os.listdir("ImagesUnknown") if f.endswith('.jpg')]) + 1
                                unknown_path = f"ImagesUnknown/unknown_{unknown_count}.jpg"
                                try:
                                    cv2.imwrite(unknown_path, face_region)
                                except cv2.error as e:
                                    self.log_attendance_message(f"Warning: Could not save unknown face: {e}")
                            
            stats_text = f"FaceNet Tracking - Today: {len(attendance_today)} people"
            cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('FaceNet Face Recognition Attendance', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cam.release()
        cv2.destroyAllWindows()
        self.attendance_active = False
        self.log_attendance_message(f"FaceNet tracking stopped. Total attendance: {len(attendance_today)}")
    
    def view_attendance(self):
        """View attendance records"""
        attendance_file = 'attendance.csv'
        if os.path.exists(attendance_file):
            try:
                df = pd.read_csv(attendance_file)
                if not df.empty:
                    recent_records = df.tail(20)
                    
                    attendance_window = tk.Toplevel(self.root)
                    attendance_window.title("Recent Attendance Records")
                    attendance_window.geometry("600x400")
                    
                    text_frame = tk.Frame(attendance_window)
                    text_frame.pack(fill='both', expand=True, padx=10, pady=10)
                    
                    scrollbar = tk.Scrollbar(text_frame)
                    scrollbar.pack(side='right', fill='y')
                    
                    text_widget = tk.Text(text_frame, yscrollcommand=scrollbar.set)
                    text_widget.pack(side='left', fill='both', expand=True)
                    scrollbar.config(command=text_widget.yview)
                    
                    text_widget.insert('1.0', recent_records.to_string(index=False))
                    text_widget.config(state='disabled')
                    
                else:
                    messagebox.showinfo("Info", "No attendance records found")
            except Exception as e:
                messagebox.showerror("Error", f"Error reading attendance file: {str(e)}")
        else:
            messagebox.showinfo("Info", "No attendance file found")
    
    # ==================== EMOTION DETECTION METHODS ====================
    
    def clear_emotion_report(self):
         self.emotion_report_text.delete(1.0, tk.END)
         self.emotion_history.clear()
         self.cognitive_state_history.clear()
    
    def start_emotion_webcam(self):
        if self.emotion_active:
            messagebox.showwarning("Warning", "Emotion analysis is already running!")
            return
        
        self.emotion_mode = 'webcam'
        self.emotion_active = True
        self.reset_emotion_session_data()
        
        self.emotion_cap = cv2.VideoCapture(0)
        if not self.emotion_cap.isOpened():
            for i in range(1, 4):
                self.emotion_cap = cv2.VideoCapture(i)
                if self.emotion_cap.isOpened():
                    break
        
        if not self.emotion_cap or not self.emotion_cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam.")
            self.emotion_active = False
            return
            
        threading.Thread(target=self.emotion_webcam_loop, daemon=True).start()
    
    def select_emotion_video(self):
        if self.emotion_active:
            messagebox.showwarning("Warning", "Emotion analysis is already running!")
            return
            
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if video_path:
            self.emotion_mode = 'video'
            self.emotion_active = True
            self.reset_emotion_session_data()
            self.emotion_video_path = video_path
            
            threading.Thread(target=self.process_emotion_video, daemon=True).start()
    
    def select_emotion_folder(self):
        if self.emotion_active:
            messagebox.showwarning("Warning", "Emotion analysis is already running!")
            return
            
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.emotion_mode = 'folder'
            self.emotion_active = True
            os.makedirs("analysis_reports", exist_ok=True)
            threading.Thread(target=self.process_emotion_folder, args=(folder_path,), daemon=True).start()
    
    def reset_emotion_session_data(self):
        """Reset emotion analysis session data"""
        self.cognitive_states_report = []
        self.emotions_report = []
        self.emotion_history.clear()
        self.cognitive_state_history.clear()
        self.total_frames_processed = 0
        self.session_start_time = datetime.datetime.now()
    
    def calculate_ear_emotion(self, eye_landmarks):
        try:
            if len(eye_landmarks) != 6:
                return 0
            
            v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
            v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
            h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
            return (v1 + v2) / (2.0 * h) if h != 0 else 0
        except Exception:
            return 0
    
    def get_emotion_cognitive_state(self, landmarks):
        try:
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            nose_tip_idx = 0
            face_width_lm_left = 33
            face_width_lm_right = 263
            
            max_required_idx = max(left_eye_indices + right_eye_indices + [nose_tip_idx, face_width_lm_left, face_width_lm_right])
            if len(landmarks) <= max_required_idx:
                return 'N/A (Lm Insufficient)'

            left_eye = [landmarks[i] for i in left_eye_indices]
            right_eye = [landmarks[i] for i in right_eye_indices]
            
            avg_ear = (self.calculate_ear_emotion(left_eye) + self.calculate_ear_emotion(right_eye)) / 2.0

            if avg_ear < 0.2:
                return 'Drowsy'
            
            nose_tip = landmarks[nose_tip_idx]
            face_width = abs(landmarks[face_width_lm_left][0] - landmarks[face_width_lm_right][0])
            
            if face_width == 0:
                return 'Attentive'

            eye_center_x = (landmarks[face_width_lm_left][0] + landmarks[face_width_lm_right][0]) / 2.0
            gaze_offset = (nose_tip[0] - eye_center_x) / face_width
            
            return 'Attentive' if abs(gaze_offset) < 0.1 else 'Distracted'
        except IndexError:
            return 'N/A (Lm Idx Error)'
        except Exception:
            return 'N/A (Cog Error)'
    
    def get_emotion_from_face(self, face_img):
        try:
            if self.emotion_model is None:
                return 'N/A (No Model)'
                
            if face_img.size == 0:
                return 'N/A (No Face)'
                
            face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img_resized = cv2.resize(face_img_gray, (48, 48))
            face_img_normalized = face_img_resized / 255.0
            face_img_expanded = np.expand_dims(face_img_normalized, axis=[0, -1])
            pred = self.emotion_model.predict(face_img_expanded, verbose=0)
            return self.emotion_labels[np.argmax(pred)]
        except cv2.error:
            return 'N/A (CV2 Error)'
        except Exception:
            return 'N/A (Emo Error)'
    
    def process_emotion_frame(self, frame, current_time=0, total_time=0):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            current_emotion_to_display = Counter(self.emotion_history).most_common(1)[0][0] if self.emotion_history else 'N/A'
            current_cognitive_to_display = Counter(self.cognitive_state_history).most_common(1)[0][0] if self.cognitive_state_history else 'N/A'
            face_detected_this_frame = False

            if results and results.multi_face_landmarks:
                face_detected_this_frame = True
                for face_landmarks_mp in results.multi_face_landmarks:
                    landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                                 for lm in face_landmarks_mp.landmark]
                    
                    if not landmarks or len(landmarks) < 388:
                        continue

                    x_coords, y_coords = zip(*landmarks)
                    x_min, x_max = max(0, min(x_coords) - 20), min(frame.shape[1], max(x_coords) + 20)
                    y_min, y_max = max(0, min(y_coords) - 20), min(frame.shape[0], max(y_coords) + 20)

                    if x_min < x_max and y_min < y_max:
                        face_img = frame[y_min:y_max, x_min:x_max]
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        emotion_raw = self.get_emotion_from_face(face_img)
                        cognitive_state_raw = self.get_emotion_cognitive_state(landmarks)
                        
                        if not (cognitive_state_raw == 'N/A' or cognitive_state_raw.startswith('N/A')):
                            self.cognitive_states_report.append(cognitive_state_raw)
                            self.cognitive_state_history.append(cognitive_state_raw)
                        if not (emotion_raw == 'N/A' or emotion_raw.startswith('N/A')):
                            self.emotions_report.append(emotion_raw)
                            self.emotion_history.append(emotion_raw)
                        
                        current_emotion_to_display = Counter(self.emotion_history).most_common(1)[0][0] if self.emotion_history else emotion_raw
                        current_cognitive_to_display = Counter(self.cognitive_state_history).most_common(1)[0][0] if self.cognitive_state_history else cognitive_state_raw
                        
                        self.total_frames_processed += 1
            
            if not face_detected_this_frame:
                current_emotion_to_display = "No face detected"
                current_cognitive_to_display = "Distracted"
                
                self.cognitive_state_history.append("Distracted")
                self.cognitive_states_report.append("Distracted")
            
            # Draw analysis results with enhanced info for video mode
            text_bg_h = 130 if self.emotion_mode in ['video', 'folder'] else 100
            text_bg_w = 500
            
            if frame.shape[0] > 10 + text_bg_h and frame.shape[1] > 10 + text_bg_w:
                overlay_area = frame[10 : 10 + text_bg_h, 10 : 10 + text_bg_w]
                text_bg = np.zeros_like(overlay_area, dtype=np.uint8)
                alpha = 0.6
                blended_overlay = cv2.addWeighted(overlay_area, 1 - alpha, text_bg, alpha, 0)
                frame[10 : 10 + text_bg_h, 10 : 10 + text_bg_w] = blended_overlay

            cv2.putText(frame, f'Emotion: {current_emotion_to_display}', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Cognitive: {current_cognitive_to_display}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            if self.emotion_mode in ['video', 'folder']:
                # Format time display
                current_min, current_sec = divmod(int(current_time), 60)
                total_min, total_sec = divmod(int(total_time), 60)
                time_str = f'Time: {current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}'
                cv2.putText(frame, time_str, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                
                # Control instructions
                if self.emotion_mode == 'video':
                    cv2.putText(frame, 'Controls: Q=Quit, A=Back 10s, D=Forward 10s', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                else:  # folder mode
                    cv2.putText(frame, 'Controls: Q=Next Video, A=Back 10s, D=Forward 10s', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Q:Quit Analysis', (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            return frame
        except Exception as e:
            return frame
    
    def emotion_webcam_loop(self):
        try:
            window_title = "Emotion Analysis - Press Q to Quit"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            
            while self.emotion_active and self.emotion_cap and self.emotion_cap.isOpened():
                ret, frame = self.emotion_cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_emotion_frame(frame.copy())
                cv2.imshow(window_title, processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.emotion_active = False
                    break
            
        except Exception as e:
            messagebox.showerror("Error", f"Webcam loop error: {e}")
        finally:
            self.finalize_emotion_session()
    
    def process_emotion_video(self):
        if not hasattr(self, 'emotion_video_path'):
            return
            
        try:
            self.emotion_cap = cv2.VideoCapture(self.emotion_video_path)
            if not self.emotion_cap.isOpened():
                messagebox.showerror("Error", f"Failed to open video")
                return
            
            # Get video properties
            fps = self.emotion_cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.emotion_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = total_frames / fps if fps > 0 else 0
            
            window_title = f"Video Analysis - Press Q to Quit"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            
            while self.emotion_active and self.emotion_cap.isOpened():
                ret, frame = self.emotion_cap.read()
                if not ret:
                    break
                
                # Get current frame position and time
                current_frame = int(self.emotion_cap.get(cv2.CAP_PROP_POS_FRAMES))
                current_time = current_frame / fps if fps > 0 else 0
                
                processed_frame = self.process_emotion_frame(frame.copy(), current_time, total_duration)
                cv2.imshow(window_title, processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.emotion_active = False
                    break
                elif key == ord('d'):  # Forward 10 seconds
                    new_frame = min(current_frame + int(10 * fps), total_frames - 1)
                    self.emotion_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                elif key == ord('a'):  # Backward 10 seconds
                    new_frame = max(current_frame - int(10 * fps), 0)
                    self.emotion_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            
        except Exception as e:
            messagebox.showerror("Error", f"Video processing error: {e}")
        finally:
            if self.emotion_cap:
                self.emotion_cap.release()
            cv2.destroyAllWindows()
            self.finalize_emotion_session()
    
    def process_emotion_folder(self, folder_path):
        video_files = []
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for f_name in sorted(os.listdir(folder_path)):
            if any(f_name.lower().endswith(ext) for ext in valid_extensions):
                video_files.append(os.path.join(folder_path, f_name))

        if not video_files:
            messagebox.showinfo("Info", "No video files found in the selected folder.")
            return

        video_index = 0
        while video_index < len(video_files) and self.emotion_active:
            video_path = video_files[video_index]
            self.reset_emotion_session_data()
            self.emotion_video_path = video_path
            
            try:
                self.emotion_cap = cv2.VideoCapture(video_path)
                if not self.emotion_cap.isOpened():
                    video_index += 1
                    continue
                
                # Get video properties
                fps = self.emotion_cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(self.emotion_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_duration = total_frames / fps if fps > 0 else 0
                
                video_filename = os.path.basename(video_path)
                window_title = f"Folder Analysis - {video_filename} ({video_index + 1}/{len(video_files)})"
                cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                
                video_finished = False
                skip_to_next = False
                
                while self.emotion_active and self.emotion_cap.isOpened() and not skip_to_next:
                    ret, frame = self.emotion_cap.read()
                    if not ret:
                        video_finished = True
                        break
                    
                    # Get current frame position and time
                    current_frame = int(self.emotion_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    current_time = current_frame / fps if fps > 0 else 0
                    
                    processed_frame = self.process_emotion_frame(frame.copy(), current_time, total_duration)
                    cv2.imshow(window_title, processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        skip_to_next = True
                        break
                    elif key == ord('d'):  # Forward 10 seconds
                        new_frame = min(current_frame + int(10 * fps), total_frames - 1)
                        self.emotion_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                    elif key == ord('a'):  # Backward 10 seconds
                        new_frame = max(current_frame - int(10 * fps), 0)
                        self.emotion_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                
                # Generate and save report for this video
                report_content = self.generate_emotion_report()
                report_filename = f"{os.path.splitext(video_filename)[0]}_analysis_report.txt"
                report_save_path = os.path.join("analysis_reports", report_filename)
                
                try:
                    with open(report_save_path, "w", encoding="utf-8") as f:
                        f.write(report_content)
                except Exception as e:
                    messagebox.showerror("Save Error", f"Failed to save report: {e}")
                
                if self.emotion_cap:
                    self.emotion_cap.release()
                cv2.destroyAllWindows()
                
                video_index += 1
                
            except Exception as e:
                messagebox.showerror("Error", f"Error processing video {video_filename}: {e}")
                video_index += 1

        self.emotion_active = False
        messagebox.showinfo("Info", "Folder analysis completed.")
    
    def finalize_emotion_session(self):
        if self.emotion_cap:
            self.emotion_cap.release()
        cv2.destroyAllWindows()
        self.emotion_active = False
        
        if self.emotion_mode in ['webcam', 'video']:
            report_content = self.generate_emotion_report()
            self.display_emotion_report(report_content)
    
    def generate_emotion_report(self):
        if self.session_start_time:
            session_duration = datetime.datetime.now() - self.session_start_time
            duration_str = str(session_duration).split('.')[0]
        else:
            duration_str = "Unknown"

        report = f"""
{'='*80}
                               EMOTION ANALYSIS REPORT
{'='*80}

Session Information:
  • Analysis Mode: {self.emotion_mode.upper() if hasattr(self, 'emotion_mode') else 'Unknown'}
  • Session Duration: {duration_str}
  • Total Frames Analyzed: {self.total_frames_processed:,}
  • Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
                                     COGNITIVE STATE ANALYSIS
{'='*80}
"""
        if self.cognitive_states_report:
            state_counts = Counter(self.cognitive_states_report)
            total_cognitive = len(self.cognitive_states_report)
            report += f"Total Cognitive State Detections: {total_cognitive:,}\n\n"
            
            for state in sorted(set(self.cognitive_labels + list(state_counts.keys()))):
                count = state_counts.get(state, 0)
                percentage = (count / total_cognitive * 100) if total_cognitive > 0 else 0
                bar_length = int(percentage / 2)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                report += f"  {state:12} │ {count:6,} │ {percentage:6.2f}% │ {bar}\n"
                
            dominant_cognitive = state_counts.most_common(1)[0][0] if state_counts else "None"
            report += f"\nDominant Cognitive State: {dominant_cognitive}\n"
        else:
            report += "No cognitive states detected.\n"

        report += f"\n{'='*80}\n"
        report += "                                     EMOTION ANALYSIS\n"
        report += f"{'='*80}\n"

        if self.emotions_report:
            emotion_counts = Counter(self.emotions_report)
            total_emotions = len(self.emotions_report)
            report += f"Total Emotion Detections: {total_emotions:,}\n\n"
            
            for emotion in sorted(set(self.emotion_labels + list(emotion_counts.keys()))):
                count = emotion_counts.get(emotion, 0)
                percentage = (count / total_emotions * 100) if total_emotions > 0 else 0
                bar_length = int(percentage / 2)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                report += f"  {emotion:12} │ {count:6,} │ {percentage:6.2f}% │ {bar}\n"
                
            dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "None"
            report += f"\nDominant Emotion: {dominant_emotion}\n"
        else:
            report += "No emotions detected.\n"

        report += f"\n{'='*80}\n"
        report += "                                     INSIGHTS & SUMMARY\n"
        report += f"{'='*80}\n"
        
        if self.cognitive_states_report:
            cog_counts = Counter(self.cognitive_states_report)
            total_valid_cog = sum(cog_counts[s] for s in self.cognitive_labels if s in cog_counts)
            if total_valid_cog > 0:
                attentive_percentage = (cog_counts.get('Attentive', 0) / total_valid_cog * 100)
                report += f"  • Attention Level: {attentive_percentage:.1f}%\n"
                if attentive_percentage > 70:
                    report += "  • Assessment: High focus and engagement levels detected.\n"
                elif attentive_percentage > 40:
                    report += "  • Assessment: Moderate focus with some distraction periods.\n"
                else:
                    report += "  • Assessment: Low focus levels, potential for disengagement.\n"

        if self.emotions_report:
            emo_counts = Counter(self.emotions_report)
            total_valid_emo = sum(emo_counts[s] for s in self.emotion_labels if s in emo_counts)
            if total_valid_emo > 0:
                positive_emotions = ['Happy', 'Surprise', 'Neutral']
                positive_count = sum(emo_counts.get(emotion, 0) for emotion in positive_emotions)
                positive_percentage = (positive_count / total_valid_emo * 100)
                report += f"  • Positive Emotional State: {positive_percentage:.1f}%\n"

        report += f"\n{'='*80}\n"
        report += "                                         END OF REPORT\n"
        report += f"{'='*80}\n\n"
        return report
    
    def display_emotion_report(self, report_content):
        self.emotion_report_text.delete(1.0, tk.END)
        self.emotion_report_text.insert(tk.END, report_content)
        self.emotion_report_text.see(1.0)
    
    # ==================== EXAM PROCTORING METHODS ====================
    
    def log_proctoring_message(self, message):
        """Add message to proctoring log display"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.proctoring_log_text.insert(tk.END, log_entry)
        self.proctoring_log_text.see(tk.END)
        
        if self.proctoring_active and "Warning" in message:
            self.violation_log.append({
                'timestamp': datetime.datetime.now(),
                'violation': message,
                'student': self.student_name
            })
    
    def start_proctoring_screening(self):
        """Start the pre-exam screening process"""
        self.log_proctoring_message("Starting pre-exam screening...")
        
        self.screening_window = tk.Toplevel(self.root)
        self.screening_window.title("Pre-Exam Screening")
        self.screening_window.geometry("800x600")
        
        instructions = tk.Label(self.screening_window, 
                               text="Please look at the camera for identification.\n"
                                    "The system will verify your identity before allowing you to take the exam.",
                               font=('Arial', 12))
        instructions.pack(pady=20)
        
        self.screening_frame = tk.Label(self.screening_window)
        self.screening_frame.pack(pady=10)
        
        self.screening_status = tk.Label(self.screening_window, 
                                        text="Status: Waiting for face detection...",
                                        font=('Arial', 12))
        self.screening_status.pack(pady=10)
        
        self.screening_cap = cv2.VideoCapture(0)
        if not self.screening_cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot access camera for screening")
            self.screening_window.destroy()
            return
            
        self.screening_active = True
        self.screening_countdown = 15
        self.screening_status.config(text=f"Status: Looking for registered student ({self.screening_countdown}s remaining)")
        
        self.screening_window.protocol("WM_DELETE_WINDOW", self.cancel_proctoring_screening)
        self.proctoring_screening_loop()
    
    def cancel_proctoring_screening(self):
        """Cancel the pre-exam screening"""
        self.screening_active = False
        if self.screening_cap:
            self.screening_cap.release()
        self.screening_window.destroy()
        self.log_proctoring_message("Pre-exam screening cancelled")
    
    def proctoring_screening_loop(self):
        """Loop for pre-exam screening"""
        if not self.screening_active:
            return
            
        try:
            ret, frame = self.screening_cap.read()
            if not ret:
                self.screening_status.config(text="Error: Failed to capture frame")
                self.screening_window.after(100, self.proctoring_screening_loop)
                return
                
            self.screening_countdown -= 0.03
            
            display_frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            self.screening_frame.configure(image=photo)
            self.screening_frame.image = photo
            
            student_id, confidence = self.recognize_face_facenet_proctoring(frame)
            
            if student_id is not None and confidence > 85:
                name = self.name_mapping.get(student_id, "Unknown")
                self.student_name = name
                self.student_id = student_id
                self.screening_active = False
                
                self.screening_cap.release()
                self.screening_window.destroy()
                
                self.start_proctoring_monitoring()
                
                self.log_proctoring_message(f"Student verified: {name} (ID: {student_id}, Confidence: {confidence:.1f}%)")
                self.log_proctoring_message("Proceeding with exam monitoring...")
                return
            
            if self.screening_countdown > 0:
                self.screening_status.config(text=f"Status: Looking for registered student ({int(self.screening_countdown)}s remaining)")
            else:
                self.screening_active = False
                self.screening_cap.release()
                self.screening_window.destroy()
                messagebox.showerror("Screening Failed", 
                                    "Could not verify your identity. Only registered students can take the exam.")
                self.log_proctoring_message("Pre-exam screening failed: Student not identified")
                
            self.screening_window.after(30, self.proctoring_screening_loop)
            
        except Exception as e:
            self.log_proctoring_message(f"Screening error: {str(e)}")
            self.cancel_proctoring_screening()
    
    def recognize_face_facenet_proctoring(self, frame):
        """Recognize face using FaceNet embeddings for proctoring"""
        try:
            if self.known_embeddings is None or self.name_mapping is None:
                return None, 0
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            boxes, probs = self.mtcnn.detect(pil_image)
            
            if boxes is not None and len(boxes) > 0:
                best_face_idx = np.argmax(probs)
                best_box = boxes[best_face_idx]
                best_prob = probs[best_face_idx]
                
                if best_prob > 0.9:
                    face_tensor = self.mtcnn.extract(pil_image, [best_box], save_path=None)
                    
                    if face_tensor is not None:
                        face_tensor = face_tensor[0].unsqueeze(0)
                        
                        with torch.no_grad():
                            embedding = self.resnet(face_tensor).numpy().flatten()
                            embedding = embedding / np.linalg.norm(embedding)
                        
                        best_match_id = None
                        best_similarity = -1
                        
                        for student_id, stored_embedding in self.known_embeddings.items():
                            similarity = np.dot(embedding, stored_embedding)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match_id = student_id
                        
                        confidence = best_similarity * 100
                        return best_match_id, confidence
            
            return None, 0
        except Exception as e:
            self.log_proctoring_message(f"Recognition error: {str(e)}")
            return None, 0
    
    def start_proctoring_monitoring(self):
        """Start the monitoring process"""
        try:
            self.proctoring_cap = cv2.VideoCapture(0)
            if not self.proctoring_cap.isOpened():
                messagebox.showerror("Camera Error", "Cannot access camera")
                return
            
            self.proctoring_active = True
            self.proctoring_session_start_time = datetime.datetime.now()
            self.violation_log = []
            self.cognitive_states = {'attentive': 0, 'distracted': 0, 'drowsy': 0, 'absent': 0}
            self.proctoring_total_frames = 0
            self.proctoring_cognitive_history.clear()
            
            self.proctoring_start_btn.config(state='disabled')
            self.proctoring_stop_btn.config(state='normal')
            self.proctoring_status_label.config(text="Status: Monitoring Active")
            
            self.proctoring_student_label.config(text=f"Student: {self.student_name}")
            
            self.log_proctoring_message("Monitoring session started")
            
            self.proctoring_monitor_thread = threading.Thread(target=self.proctoring_monitor_loop, daemon=True)
            self.proctoring_monitor_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")
    
    def stop_proctoring_monitoring(self):
        """Stop the monitoring process"""
        self.proctoring_active = False
        
        if self.proctoring_cap:
            self.proctoring_cap.release()
        
        self.proctoring_start_btn.config(state='normal')
        self.proctoring_stop_btn.config(state='disabled')
        self.proctoring_status_label.config(text="Status: Monitoring Stopped")
        
        self.log_proctoring_message("Monitoring session ended")
    
    def proctoring_monitor_loop(self):
        """Main monitoring loop"""
        while self.proctoring_active:
            try:
                ret, frame = self.proctoring_cap.read()
                if not ret:
                    break
                
                self.proctoring_total_frames += 1
                
                processed_frame = self.process_proctoring_frame(frame.copy())
                
                self.update_proctoring_video_display(processed_frame)
                
                time.sleep(0.03)
                
            except Exception as e:
                self.log_proctoring_message(f"Error in monitoring loop: {str(e)}")
                break
    
    def process_proctoring_frame(self, frame):
        """Process each frame for violations and cognitive states"""
        faces = self.detect_proctoring_face_roi(frame)
        
        self.analyze_proctoring_face_and_cognitive_state(frame, faces)
        
        if self.yolo_model is not None:
            self.detect_banned_objects_proctoring(frame)
            self.count_persons_proctoring(frame)
        
        return frame
    
    def detect_proctoring_face_roi(self, frame):
        """Detect face and return ROI with landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        faces = []
        if results.multi_face_landmarks:
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                h, w = frame.shape[:2]
                x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
                
                x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
                y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)
                
                face_roi = frame[y_min:y_max, x_min:x_max]
                
                faces.append({
                    'roi': face_roi,
                    'landmarks': face_landmarks,
                    'bbox': (x_min, y_min, x_max, y_max)
                })
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return faces
    
    def calculate_ear_proctoring(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for drowsiness detection"""
        try:
            if len(eye_landmarks) != 6:
                return 0.3
            
            vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
            vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
            horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
            
            if horizontal == 0:
                return 0.3
            
            return (vertical_1 + vertical_2) / (2.0 * horizontal)
        except Exception as e:
            return 0.3
    
    def get_proctoring_cognitive_state(self, landmarks, w, h):
        """Determine cognitive state based on landmarks"""
        try:
            pixel_landmarks = []
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                pixel_landmarks.append((x, y))
            
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            nose_tip_idx = 1
            face_width_lm_left = 33
            face_width_lm_right = 263
            
            max_required_idx = max(left_eye_indices + right_eye_indices + 
                                 [nose_tip_idx, face_width_lm_left, face_width_lm_right])
            if len(pixel_landmarks) <= max_required_idx:
                return 'absent'
            
            left_eye = [pixel_landmarks[i] for i in left_eye_indices]
            right_eye = [pixel_landmarks[i] for i in right_eye_indices]
            
            avg_ear = (self.calculate_ear_proctoring(left_eye) + self.calculate_ear_proctoring(right_eye)) / 2.0
            
            if avg_ear < 0.2:
                return 'drowsy'
            
            nose_tip = pixel_landmarks[nose_tip_idx]
            face_width = abs(pixel_landmarks[face_width_lm_left][0] - 
                            pixel_landmarks[face_width_lm_right][0])
            
            if face_width == 0:
                return 'attentive'
                
            eye_center_x = (pixel_landmarks[face_width_lm_left][0] + 
                           pixel_landmarks[face_width_lm_right][0]) / 2.0
            gaze_offset = (nose_tip[0] - eye_center_x) / face_width
            
            return 'attentive' if abs(gaze_offset) < 0.1 else 'distracted'
            
        except IndexError:
            return 'absent'
        except Exception as e:
            return 'attentive'
    
    def analyze_proctoring_face_and_cognitive_state(self, frame, faces):
        """Analyze face recognition and cognitive states"""
        if not faces:
            self.cognitive_states['absent'] += 1
            self.proctoring_total_frames += 1
            
            if self.proctoring_total_frames % 15 == 0:
                self.proctoring_cognitive_label.config(text="State: No Face Detected")
                if self.proctoring_total_frames % 45 == 0:
                    self.log_proctoring_message("Warning: No face detected - student may have left")
            return
        
        if len(faces) > 1:
            self.log_proctoring_message("Warning: Multiple faces detected - possible cheating")
        
        face_data = faces[0]
        landmarks = face_data['landmarks'].landmark
        h, w = frame.shape[:2]
        
        current_state = self.get_proctoring_cognitive_state(landmarks, w, h)
        
        self.proctoring_cognitive_history.append(current_state)
        
        most_common_state = Counter(self.proctoring_cognitive_history).most_common(1)[0][0]
        
        self.cognitive_states[most_common_state] += 1
        self.proctoring_total_frames += 1
        
        if most_common_state == 'drowsy':
            self.proctoring_cognitive_label.config(text="State: Drowsy (Eyes Closed)")
            if self.proctoring_total_frames % 15 == 0:
                self.log_proctoring_message("Warning: Student appears drowsy (eyes closed)")
        elif most_common_state == 'distracted':
            self.proctoring_cognitive_label.config(text="State: Distracted (Looking Away)")
            if self.proctoring_total_frames % 20 == 0:
                self.log_proctoring_message("Warning: Student looking away from screen")
        else:
            self.proctoring_cognitive_label.config(text="State: Attentive")
        
        self.visualize_proctoring_cognitive_state(frame, face_data['bbox'], most_common_state)
    
    def visualize_proctoring_cognitive_state(self, frame, bbox, state):
        """Visualize cognitive state on the video frame"""
        x_min, y_min, x_max, y_max = bbox
        
        if state == 'drowsy':
            color = (0, 0, 255)
            cv2.putText(frame, "DROWSY (EYES CLOSED)", (x_min, y_min - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif state == 'distracted':
            color = (0, 165, 255)
            cv2.putText(frame, "DISTRACTED", (x_min, y_min - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            color = (0, 255, 0)
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    
    def detect_banned_objects_proctoring(self, frame):
        """Detect banned objects using YOLOv8"""
        try:
            results = self.yolo_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id in self.banned_objects and confidence > 0.5:
                            object_name = self.banned_objects[class_id]
                            self.log_proctoring_message(f"Warning: Banned object detected - {object_name}")
                            
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{object_name}", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except Exception as e:
            pass
    
    def count_persons_proctoring(self, frame):
        """Count number of persons in frame"""
        try:
            results = self.yolo_model(frame, verbose=False)
            person_count = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id == 0 and confidence > 0.5:
                            person_count += 1
            
            if person_count > 1:
                self.log_proctoring_message(f"Warning: {person_count} persons detected - possible cheating")
                
        except Exception as e:
            pass
    
    def update_proctoring_video_display(self, frame):
        """Update the video display in proctoring GUI"""
        try:
            display_frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.proctoring_video_label.configure(image=photo)
            self.proctoring_video_label.image = photo
            
            self.proctoring_violations_label.config(text=f"Violations: {len(self.violation_log)}")
            
        except Exception as e:
            pass
    
    def generate_proctoring_report(self):
        """Generate detailed monitoring report"""
        try:
            if not self.violation_log and self.proctoring_total_frames == 0:
                messagebox.showwarning("No Data", "No monitoring data available to generate report")
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                title="Save Monitoring Report"
            )
            
            if not filename:
                return
            
            report_content = self.create_proctoring_report_content()
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            messagebox.showinfo("Report Generated", f"Monitoring report saved to: {filename}")
            self.log_proctoring_message(f"Report generated: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def create_proctoring_report_content(self):
        """Create detailed report content"""
        report = []
        report.append("=" * 60)
        report.append("AI-POWERED EXAM PROCTORING SYSTEM - MONITORING REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Session Information
        report.append("SESSION INFORMATION:")
        report.append("-" * 30)
        if self.proctoring_session_start_time:
            report.append(f"Session Start Time: {self.proctoring_session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            if not self.proctoring_active:
                duration = datetime.datetime.now() - self.proctoring_session_start_time
                report.append(f"Session Duration: {str(duration).split('.')[0]}")
        report.append(f"Student Name: {self.student_name}")
        report.append(f"Student ID: {self.student_id}")
        report.append(f"Total Frames Processed: {self.proctoring_total_frames}")
        report.append(f"Total Violations: {len(self.violation_log)}")
        report.append("")
        
        # Cognitive State Analysis
        if self.proctoring_total_frames > 0:
            report.append("COGNITIVE STATE ANALYSIS:")
            report.append("-" * 30)
            attentive_pct = (self.cognitive_states['attentive'] / self.proctoring_total_frames) * 100
            distracted_pct = (self.cognitive_states['distracted'] / self.proctoring_total_frames) * 100
            drowsy_pct = (self.cognitive_states['drowsy'] / self.proctoring_total_frames) * 100
            absent_pct = (self.cognitive_states['absent'] / self.proctoring_total_frames) * 100
            
            report.append(f"Attentive: {attentive_pct:.1f}% ({self.cognitive_states['attentive']} frames)")
            report.append(f"Distracted: {distracted_pct:.1f}% ({self.cognitive_states['distracted']} frames)")
            report.append(f"Drowsy: {drowsy_pct:.1f}% ({self.cognitive_states['drowsy']} frames)")
            report.append(f"Absent/No Face: {absent_pct:.1f}% ({self.cognitive_states['absent']} frames)")
            report.append("")
        
        # Violation Log
        if self.violation_log:
            report.append("DETAILED VIOLATION LOG:")
            report.append("-" * 30)
            for i, violation in enumerate(self.violation_log, 1):
                timestamp = violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                report.append(f"{i:3d}. [{timestamp}] {violation['violation']}")
            report.append("")
        
        # Summary and Recommendations
        report.append("SUMMARY AND RECOMMENDATIONS:")
        report.append("-" * 30)
        violation_count = len(self.violation_log)
        
        if violation_count == 0:
            report.append("✓ No violations detected during the monitoring session.")
            report.append("✓ Student behavior appears to be compliant with exam protocols.")
        elif violation_count <= 3:
            report.append("⚠ Minor violations detected. Review recommended.")
            report.append("• Consider providing additional guidance on exam protocols.")
        else:
            report.append("⚠ Multiple violations detected. Detailed review required.")
            report.append("• Recommend manual review of exam session.")
            report.append("• Consider additional proctoring measures.")
        
        if self.proctoring_total_frames > 0:
            if self.cognitive_states['absent'] / self.proctoring_total_frames > 0.1:
                report.append("⚠ Student was frequently absent from frame.")
            if self.cognitive_states['drowsy'] / self.proctoring_total_frames > 0.15:
                report.append("⚠ Student showed signs of drowsiness during exam.")
        
        report.append("")
        report.append("Report generated on: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        report.append("=" * 60)
        
        return "\n".join(report)
    
    # ==================== MAIN APPLICATION METHODS ====================
    
    def update_status(self, message):
        """Update the status bar"""
        self.status_bar.config(text=message)
    
    def on_closing(self):
        """Handle application closing"""
        # Stop all active processes
        self.attendance_active = False
        self.emotion_active = False
        self.proctoring_active = False
        self.screening_active = False
        
        # Release cameras
        if self.attendance_cap:
            self.attendance_cap.release()
        if self.emotion_cap:
            self.emotion_cap.release()
        if self.proctoring_cap:
            self.proctoring_cap.release()
        if self.screening_cap:
            self.screening_cap.release()
        
        # Close face mesh
        if hasattr(self, 'face_mesh') and self.face_mesh:
            try:
                self.face_mesh.close()
            except:
                pass
        
        # Destroy windows
        cv2.destroyAllWindows()
        
        # Close application
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_status("EduSense AI Ready - Select a module to begin")
        self.root.mainloop()

def main():
    """Main function to run EduSense AI"""
    try:
        app = EduSenseAI()
        app.run()
    except Exception as e:
        print(f"Error starting EduSense AI: {e}")
        messagebox.showerror("Startup Error", f"Failed to start EduSense AI: {e}")

if __name__ == "__main__":
    main()