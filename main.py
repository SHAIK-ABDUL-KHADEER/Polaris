import os
import cv2
import numpy as np
import mediapipe as mp
import pygame
import json
import math
import time
from typing import Dict, List, Tuple, Optional


class PoseSkeletonDetector:
    """Handles pose detection and dynamic skeleton rendering with improved validation"""

    def __init__(self, config: Dict):
        self.config = config

        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Camera
        self.cap = None
        self.frame = None

        # Skeleton visualization
        self.skeleton_connections = [
            # Head to shoulders
            (0, 11), (0, 12),  # nose to shoulders
            (11, 12),  # shoulder line

            # Arms
            (11, 13), (13, 15),  # left arm
            (12, 14), (14, 16),  # right arm

            # Torso
            (11, 23), (12, 24),  # shoulders to hips
            (23, 24),  # hip line

            # Legs
            (23, 25), (25, 27), (27, 31),  # left leg
            (24, 26), (26, 28), (28, 32),  # right leg
        ]

        # Color animation
        self.color_phase = 0
        self.movement_intensity = 0
        self.prev_landmarks = None

        # Critical landmarks for improved validation (from 3D avatar system)
        self.critical_landmarks = [
            # Head
            self.mp_pose.PoseLandmark.NOSE,
            # Shoulders
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            # Hips
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            # Feet
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
        ]

        # Validation debug counter
        self._validation_debug_counter = 0

    def initialize_camera(self) -> bool:
        """Initialize camera with error handling"""
        try:
            self.cap = cv2.VideoCapture(self.config['camera_index'])
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return False

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            print("Camera initialized successfully")
            return True

        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False

    def detect_pose(self) -> Tuple[Optional[np.ndarray], str]:
        """
        Detect pose and return landmarks with status message - improved validation
        Returns: (landmarks, status_message)
        """
        if not self.cap or not self.cap.isOpened():
            return None, "Camera not available"

        ret, frame = self.cap.read()
        if not ret:
            return None, "Could not read from camera"

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        self.frame = frame.copy()

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks is None:
            return None, "No person detected - Please stand in front of camera"

        landmarks = results.pose_landmarks.landmark

        # Use improved validation from 3D avatar system
        validation_failed = self.validate_full_body_pose(landmarks, frame.shape)
        if validation_failed:
            return None, "Please step back and ensure full body is visible"

        # Check if person is at appropriate distance (original logic)
        distance_status = self.check_distance_legacy(landmarks, frame.shape)
        if distance_status != "OK":
            return landmarks, distance_status

        # Calculate movement for color animation
        self.update_movement_intensity(landmarks)

        return landmarks, "Pose detected"

    def validate_full_body_pose(self, landmarks, frame_shape) -> bool:
        """
        Improved validation from 3D avatar system
        Check if all critical body parts are visible in frame
        Returns True if validation FAILED (person should go back)
        """
        height, width = frame_shape[:2]

        # More lenient validation for better user experience
        missing_parts = []

        for landmark_idx in self.critical_landmarks:
            landmark = landmarks[landmark_idx.value]  # Use .value for enum

            # Check visibility score - be more lenient
            if landmark.visibility < 0.3:  # Reduced from 0.5
                missing_parts.append(f"landmark_{landmark_idx.value}")
                continue

            # Check if landmark is within frame bounds (with more generous margin)
            margin = 0.1  # Increased from 0.05 to 0.1 (10% margin)
            x, y = landmark.x, landmark.y

            if (x < -margin or x > (1 + margin) or
                    y < -margin or y > (1 + margin)):
                missing_parts.append(f"landmark_{landmark_idx.value}_out_of_bounds")

        # Only show "go back" if more than 2 critical parts are missing/invalid
        if len(missing_parts) > 2:
            self._validation_debug_counter += 1

            # Debug output every 30 frames
            if self._validation_debug_counter % 30 == 0:
                print(f"Validation failed: {len(missing_parts)} issues - {missing_parts[:3]}")
            return True

        return False

    def check_distance_legacy(self, landmarks, frame_shape) -> str:
        """Original distance checking logic (keep as backup)"""
        height, width = frame_shape[:2]

        # Get key landmark indices
        distance_landmarks = [11, 12, 23, 24]  # shoulders and hips

        # Calculate average size of key landmarks
        if not all(idx < len(landmarks) for idx in distance_landmarks):
            return "OK"

        # Get shoulder and hip positions
        positions = []
        for idx in distance_landmarks:
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                positions.append([lm.x, lm.y])

        if len(positions) < 3:
            return "Move to be fully visible in camera"

        positions = np.array(positions)

        # Calculate bounding box
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)

        person_width = max_x - min_x
        person_height = max_y - min_y

        # Check if too close (person takes up too much of frame)
        if person_width > 0.8 or person_height > 0.9:
            return "Too close - Please step back"

        # Check if too far (person too small)
        if person_width < 0.2 or person_height < 0.3:
            return "Too far - Please step closer"

        # Check if person is centered enough
        center_x = (min_x + max_x) / 2
        if center_x < 0.2 or center_x > 0.8:
            return "Please center yourself in camera view"

        return "OK"

    def update_movement_intensity(self, landmarks):
        """Calculate movement intensity for color animation"""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            self.movement_intensity = 0
            return

        # Calculate movement of key joints
        movement_sum = 0
        joint_count = 0

        key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]  # arms, shoulders, hips, knees

        for idx in key_joints:
            if idx < len(landmarks) and idx < len(self.prev_landmarks):
                curr = landmarks[idx]
                prev = self.prev_landmarks[idx]

                if curr.visibility > 0.5 and prev.visibility > 0.5:
                    dx = curr.x - prev.x
                    dy = curr.y - prev.y
                    dz = curr.z - prev.z

                    movement = math.sqrt(dx * dx + dy * dy + dz * dz)
                    movement_sum += movement
                    joint_count += 1

        if joint_count > 0:
            self.movement_intensity = movement_sum / joint_count
            # Smooth the movement intensity
            self.movement_intensity = min(self.movement_intensity * 100, 1.0)

        self.prev_landmarks = landmarks

    def get_skeleton_color(self) -> Tuple[int, int, int]:
        """Generate dynamic color based on movement"""
        self.color_phase += 0.1

        # Base color cycling through rainbow
        base_hue = (self.color_phase % (2 * math.pi))

        # Intensity affects saturation and brightness
        saturation = 0.7 + 0.3 * self.movement_intensity
        brightness = 0.8 + 0.2 * self.movement_intensity

        # Convert HSV to RGB
        h = base_hue / (2 * math.pi)
        s = min(saturation, 1.0)
        v = min(brightness, 1.0)

        # HSV to RGB conversion
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c

        if h < 1 / 6:
            r, g, b = c, x, 0
        elif h < 2 / 6:
            r, g, b = x, c, 0
        elif h < 3 / 6:
            r, g, b = 0, c, x
        elif h < 4 / 6:
            r, g, b = 0, x, c
        elif h < 5 / 6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame"""
        return self.frame

    def cleanup(self):
        """Cleanup camera resources"""
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()


class BackgroundManager:
    """Manages background image loading and cycling"""

    def __init__(self):
        self.backgrounds = ["beach.jpg", "forest.jpg", "Graveyard.jpg", "moon.jpg"]
        self.thumbnails = []
        self.current_index = 0
        self.bg_surface = None

        # Load thumbnails
        for path in self.backgrounds:
            try:
                img = pygame.image.load(path)
                thumb = pygame.transform.scale(img, (50, 50))
                self.thumbnails.append(thumb)
            except Exception as e:
                print(f"Error loading thumbnail for {path}: {e}")
                # Placeholder thumbnail
                placeholder = pygame.Surface((50, 50))
                placeholder.fill((100, 100, 100))
                self.thumbnails.append(placeholder)

        print(f"Loaded {len(self.backgrounds)} background images")

    def load_current_background(self, display_size: Tuple[int, int]) -> bool:
        """Load and scale current background image"""
        if not self.backgrounds:
            return False

        try:
            bg_path = self.backgrounds[self.current_index]
            bg_image = pygame.image.load(bg_path)
            self.bg_surface = pygame.transform.scale(bg_image, display_size)
            return True
        except Exception as e:
            print(f"Error loading background {bg_path}: {e}")
            return False

    def select_background(self, index: int, display_size: Tuple[int, int]):
        """Select a specific background"""
        if 0 <= index < len(self.backgrounds):
            self.current_index = index
            self.load_current_background(display_size)
            print(f"Switched to background: {os.path.basename(self.backgrounds[self.current_index])}")

    def draw(self, screen):
        """Draw current background"""
        if self.bg_surface:
            screen.blit(self.bg_surface, (0, 0))
        else:
            # Fallback black background
            screen.fill((0, 0, 0))


class MusicManager:
    """Manages music loading and playback"""

    def __init__(self):
        pygame.mixer.init()
        self.musics = ["music1.mp3", "music2.mp3", "music3.mp3", "music4.mp3"]
        self.current_index = -1  # No music playing initially

    def play(self, index: int):
        """Play selected music"""
        if index == self.current_index:
            return  # Already playing

        if 0 <= index < len(self.musics):
            try:
                pygame.mixer.music.stop()
                path = self.musics[index]
                pygame.mixer.music.load(path)
                pygame.mixer.music.play(-1)  # Loop indefinitely
                self.current_index = index
                print(f"Playing music: {os.path.basename(self.musics[self.current_index])}")
            except Exception as e:
                print(f"Error playing music {path}: {e}")

    def stop(self):
        """Stop music playback"""
        pygame.mixer.music.stop()
        self.current_index = -1


class SkeletonApp:
    """Main application class"""

    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.running = False

        # Screen divisions
        self.left_width = int(self.config['window_width'] * 0.7)
        self.right_width = self.config['window_width'] - self.left_width

        # Initialize components
        self.pose_detector = PoseSkeletonDetector(self.config)
        self.background_manager = BackgroundManager()
        self.music_manager = MusicManager()

        # Display
        self.screen = None
        self.clock = None
        self.font = None

        # Buttons
        self.bg_buttons = []
        self.music_buttons = []
        self.music_text_surfaces = []

        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        # Debug counter for pose detection
        self._debug_counter = 0

    def load_config(self, path: str) -> Dict:
        """Load configuration with defaults"""
        default_config = {
            "window_width": 1024,
            "window_height": 768,
            "camera_index": 0,
            "target_fps": 30,
            "skeleton_thickness": 4,
            "joint_radius": 8,
            "status_font_size": 36,
            "show_camera_preview": True,
            "camera_preview_size": 200
        }

        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                print(f"Loaded config from {path}")
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")

        return default_config

    def initialize(self) -> bool:
        """Initialize pygame and components"""
        try:
            pygame.init()
            pygame.font.init()

            self.screen = pygame.display.set_mode(
                (self.config['window_width'], self.config['window_height'])
            )
            pygame.display.set_caption("Dynamic Pose Skeleton Detector - Improved Detection")

            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, self.config['status_font_size'])

            # Initialize camera
            if not self.pose_detector.initialize_camera():
                return False

            # Load initial background
            display_size = (self.left_width, self.config['window_height'])
            self.background_manager.load_current_background(display_size)

            # Create music text surfaces
            for i in range(4):
                surf = self.font.render(f"M{i+1}", True, (255, 255, 255))
                self.music_text_surfaces.append(surf)

            # Create buttons
            button_size = 60
            spacing = 10
            start_x = self.left_width + spacing
            start_y_themes = self.config['window_height'] - 150  # Bottom section for themes

            # Background buttons (top row in bottom section)
            for i in range(4):
                x = start_x + i * (button_size + spacing)
                rect = pygame.Rect(x, start_y_themes, button_size, button_size)
                self.bg_buttons.append((rect, i))

            # Music buttons (bottom row)
            start_y_music = start_y_themes + button_size + spacing
            for i in range(4):
                x = start_x + i * (button_size + spacing)
                rect = pygame.Rect(x, start_y_music, button_size, button_size)
                self.music_buttons.append((rect, i))

            print("Application initialized successfully")
            return True

        except Exception as e:
            print(f"Initialization error: {e}")
            return False

    def draw_skeleton(self, landmarks, status_message: str):
        """Draw skeleton with dynamic colors and proper thickness"""
        if landmarks is None:
            return

        # Get current skeleton color
        skeleton_color = self.pose_detector.get_skeleton_color()
        joint_color = (
            min(255, skeleton_color[0] + 50),
            min(255, skeleton_color[1] + 50),
            min(255, skeleton_color[2] + 50)
        )

        # Convert normalized coordinates to left panel coordinates
        panel_width = self.left_width
        panel_height = self.config['window_height']

        points = []
        for lm in landmarks:
            x = int(lm.x * panel_width)
            y = int(lm.y * panel_height)
            points.append((x, y))

        # Draw skeleton connections
        thickness = self.config['skeleton_thickness']
        for start_idx, end_idx in self.pose_detector.skeleton_connections:
            if (start_idx < len(points) and end_idx < len(points) and
                    start_idx < len(landmarks) and end_idx < len(landmarks)):

                # Only draw if both landmarks are visible
                if (landmarks[start_idx].visibility > 0.5 and
                        landmarks[end_idx].visibility > 0.5):
                    pygame.draw.line(
                        self.screen, skeleton_color,
                        points[start_idx], points[end_idx],
                        thickness
                    )

        # Draw joints
        joint_radius = self.config['joint_radius']
        key_joints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # nose, arms, legs

        for idx in key_joints:
            if idx < len(points) and idx < len(landmarks):
                if landmarks[idx].visibility > 0.5:
                    pygame.draw.circle(
                        self.screen, joint_color,
                        points[idx], joint_radius
                    )

    def draw_status(self, status_message: str):
        """Draw status message with improved color coding"""
        color = (255, 255, 255)  # white text
        if "step back" in status_message.lower():
            color = (255, 100, 100)  # red for too close
        elif "step closer" in status_message.lower():
            color = (255, 255, 100)  # yellow for too far
        elif "no person" in status_message.lower():
            color = (255, 200, 100)  # orange for no detection
        elif "full body" in status_message.lower() or "step back" in status_message.lower():
            color = (255, 150, 150)  # light red for full body visibility warning
        elif "pose detected" in status_message.lower():
            color = (100, 255, 100)  # green for good detection

        text_surface = self.font.render(status_message, True, color)
        text_rect = text_surface.get_rect()
        text_rect.centerx = self.left_width + self.right_width // 2
        text_rect.y = 50

        # Draw semi-transparent background for text
        bg_rect = text_rect.copy()
        bg_rect.inflate_ip(20, 10)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
        bg_surface.fill((0, 0, 0))
        bg_surface.set_alpha(128)
        self.screen.blit(bg_surface, bg_rect)

        # Draw text
        self.screen.blit(text_surface, text_rect)

    def draw_camera_preview(self):
        """Draw small camera preview in top right"""
        if not self.config['show_camera_preview']:
            return

        frame = self.pose_detector.get_frame()
        if frame is None:
            return

        try:
            preview_size = self.config['camera_preview_size']

            # Resize frame
            frame_resized = cv2.resize(frame, (preview_size, preview_size))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Convert to pygame surface
            preview_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

            # Position in top right, centered
            x = self.left_width + (self.right_width - preview_size) // 2
            y = 100

            self.screen.blit(preview_surface, (x, y))

            # Draw border
            pygame.draw.rect(self.screen, (255, 255, 255),
                             (x, y, preview_size, preview_size), 2)

        except Exception as e:
            print(f"Error drawing camera preview: {e}")

    def draw_fps(self):
        """Draw FPS counter in top right"""
        fps_text = f"FPS: {self.current_fps}"
        fps_surface = self.font.render(fps_text, True, (255, 255, 255))
        self.screen.blit(fps_surface, (self.left_width + 10, 10))

    def draw_buttons(self):
        """Draw theme buttons in bottom right"""
        # Background buttons
        for rect, idx in self.bg_buttons:
            thumb = self.background_manager.thumbnails[idx]
            self.screen.blit(thumb, rect.topleft)
            if self.background_manager.current_index == idx:
                pygame.draw.rect(self.screen, (255, 255, 0), rect, 3)

        # Music buttons
        for rect, idx in self.music_buttons:
            pygame.draw.rect(self.screen, (100, 100, 100), rect)
            text = self.music_text_surfaces[idx]
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
            if self.music_manager.current_index == idx:
                pygame.draw.rect(self.screen, (0, 255, 0), rect, 3)

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time

    def run(self):
        """Main application loop"""
        if not self.initialize():
            print("Failed to initialize application")
            return

        self.running = True
        print("\nControls:")
        print("- ESC: Exit")
        print("- SPACE: Toggle camera preview")
        print("- Click buttons for backgrounds and music")
        print("\nImproved Detection Features:")
        print("- Better validation for full body visibility")
        print("- More lenient pose detection thresholds")
        print("- Enhanced debug output for troubleshooting")
        print("\nApplication started successfully!")

        try:
            while self.running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_SPACE:
                            self.config['show_camera_preview'] = not self.config['show_camera_preview']
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        pos = event.pos
                        for rect, idx in self.bg_buttons:
                            if rect.collidepoint(pos):
                                display_size = (self.left_width, self.config['window_height'])
                                self.background_manager.select_background(idx, display_size)
                                break
                        for rect, idx in self.music_buttons:
                            if rect.collidepoint(pos):
                                self.music_manager.play(idx)
                                break

                # Draw background on left
                self.background_manager.draw(self.screen)

                # Detect pose
                landmarks, status = self.pose_detector.detect_pose()

                # Debug output every 60 frames (about once per 2 seconds at 30fps)
                self._debug_counter += 1
                if self._debug_counter % 60 == 0:
                    detection_status = "Valid pose detected" if (landmarks and status == "Pose detected") else f"Detection issue: {status}"
                    print(f"Pose detection status: {detection_status}")

                # Draw skeleton if pose detected and status is OK
                if landmarks and status == "Pose detected":
                    self.draw_skeleton(landmarks, status)

                # Draw right panel background
                pygame.draw.rect(self.screen, (30, 30, 30), (self.left_width, 0, self.right_width, self.config['window_height']))

                # Draw status message
                self.draw_status(status)

                # Draw camera preview
                self.draw_camera_preview()

                # Draw FPS
                self.draw_fps()

                # Draw buttons
                self.draw_buttons()

                # Update display
                pygame.display.flip()
                self.clock.tick(self.config['target_fps'])
                self.update_fps()

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.pose_detector.cleanup()
        self.music_manager.stop()
        pygame.quit()
        print("Cleanup completed")


def create_default_config():
    """Create default configuration file"""
    config = {
        "window_width": 1024,
        "window_height": 768,
        "camera_index": 0,
        "target_fps": 30,
        "skeleton_thickness": 4,
        "joint_radius": 8,
        "status_font_size": 36,
        "show_camera_preview": True,
        "camera_preview_size": 200
    }

    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    print("Created default config.json")


if __name__ == "__main__":
    print("=" * 60)
    print("   DYNAMIC POSE SKELETON DETECTOR - IMPROVED")
    print("=" * 60)
    print("Required packages: opencv-python, mediapipe, pygame, numpy")
    print("Improvements:")
    print("- Better pose validation from 3D avatar system")
    print("- More lenient detection thresholds")
    print("- Enhanced debug output for troubleshooting")
    print("=" * 60)

    # Create default files if they don't exist
    if not os.path.exists("config.json"):
        create_default_config()

    try:
        app = SkeletonApp()
        app.run()
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install with: pip install opencv-python mediapipe pygame numpy")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()