import cv2
import numpy as np
import urllib.request
import os
import time
from datetime import datetime

# Download required files if not present
face_cascade_path = "haarcascade_frontalface_default.xml"
profile_cascade_path = "haarcascade_profileface.xml"
eye_cascade_path = "haarcascade_eye.xml"

# Download Haar cascade files if needed
if not os.path.isfile(face_cascade_path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", face_cascade_path)
if not os.path.isfile(profile_cascade_path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_profileface.xml", profile_cascade_path)
if not os.path.isfile(eye_cascade_path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml", eye_cascade_path)

# Create faces directory if it doesn't exist
faces_dir = 'faces'
if not os.path.exists(faces_dir):
    os.makedirs(faces_dir)
    print(f"Created directory: {faces_dir}")
else:
    print(f"Directory already exists: {faces_dir}")

def apply_image_preprocessing(image):
    """Apply advanced image preprocessing techniques to improve detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    gray = cv2.equalizeHist(gray)
    
    # Apply gamma correction to enhance contrast
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gray = cv2.LUT(gray, table)
    
    # Apply bilateral filtering to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Create a CLAHE object for contrast-limited adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    return gray

def multi_stage_face_detection(frame, confidence_threshold=0.8):
    """
    Implements a multi-stage face detection pipeline with confidence scoring for video frames
    """
    # Apply advanced preprocessing
    processed_gray = apply_image_preprocessing(frame)
    
    # Initialize cascade classifiers
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    if face_cascade.empty() or profile_cascade.empty() or eye_cascade.empty():
        raise ValueError("Error loading cascade classifiers")
    
    # Detect faces with the primary face detector (frontal)
    frontal_faces = face_cascade.detectMultiScale(
        processed_gray,
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Also try profile face detection for completeness
    profile_faces = profile_cascade.detectMultiScale(
        processed_gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Combine all potential face candidates
    all_candidates = list(frontal_faces) + list(profile_faces)
    
    # Apply confidence scoring and verification
    verified_faces = []
    face_scores = []
    
    for (x, y, w, h) in all_candidates:
        # Start with a base confidence score
        confidence = 0.5
        
        # Extract the face ROI (Region of Interest)
        face_roi = processed_gray[y:y+h, x:x+w]
        
        # 1. Check if face is large enough (size-based confidence)
        area = w * h
        if area > 3000:
            confidence += 0.1
        
        # 2. Check aspect ratio (faces are generally more square than rectangular)
        aspect_ratio = w / h
        if 0.8 <= aspect_ratio <= 1.2:
            confidence += 0.1
        
        # 3. Check for eyes in the upper half of the face
        eyes = eye_cascade.detectMultiScale(face_roi[:h//2, :], scaleFactor=1.1, minNeighbors=3)
        if len(eyes) >= 2:
            confidence += 0.2
            
            # Eye position verification
            eye_positions_valid = True
            for (ex, ey, ew, eh) in eyes:
                # Check if eyes are in sensible positions
                if not (0.1 <= ey/h <= 0.5 and 0.05 <= ex/w <= 0.95):
                    eye_positions_valid = False
            
            if eye_positions_valid:
                confidence += 0.1
        
        # 4. Check skin color consistency
        # Convert face ROI to YCrCb color space
        if h > 0 and w > 0:
            try:
                face_color_roi = frame[y:y+h, x:x+w]
                face_ycrcb = cv2.cvtColor(face_color_roi, cv2.COLOR_BGR2YCrCb)
                cr_channel = face_ycrcb[:, :, 1]
                cb_channel = face_ycrcb[:, :, 2]
                
                # Check if values are in typical skin color range
                skin_pixels = np.logical_and(
                    np.logical_and(133 <= cr_channel, cr_channel <= 173),
                    np.logical_and(77 <= cb_channel, cb_channel <= 127)
                )
                
                skin_ratio = np.sum(skin_pixels) / (w * h)
                if skin_ratio > 0.4:
                    confidence += 0.1
            except:
                pass  # Skip color check if it fails
        
        # Check if confidence exceeds threshold
        if confidence >= confidence_threshold:
            # Check for overlaps with existing faces
            is_overlapping = False
            for i, (vx, vy, vw, vh) in enumerate(verified_faces):
                # Calculate intersection area
                intersection_width = min(x + w, vx + vw) - max(x, vx)
                intersection_height = min(y + h, vy + vh) - max(y, vy)
                
                if intersection_width > 0 and intersection_height > 0:
                    intersection_area = intersection_width * intersection_height
                    min_area = min(w * h, vw * vh)
                    overlap_ratio = intersection_area / min_area
                    
                    if overlap_ratio > 0.5:
                        is_overlapping = True
                        # Keep the one with higher confidence
                        if confidence > face_scores[i]:
                            verified_faces[i] = (x, y, w, h)
                            face_scores[i] = confidence
                        break
            
            if not is_overlapping:
                verified_faces.append((x, y, w, h))
                face_scores.append(confidence)
    
    return verified_faces, face_scores

def save_face_from_frame(frame, x, y, w, h, score, directory, frame_count):
    """Save a detected face from a video frame"""
    face_img = frame[y:y+h, x:x+w]
    
    # Create filename with timestamp, frame number and score
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_frame{frame_count}_face_score{score:.2f}_{timestamp}.jpg"
    save_path = os.path.join(directory, filename)
    
    # Save the face image
    cv2.imwrite(save_path, face_img)
    return save_path

def process_video(video_source=0, save_faces=True, display_video=True, confidence_threshold=0.75, save_interval=30):
    """
    Process video from webcam or file for face detection
    
    Args:
        video_source: 0 for webcam, or path to video file
        save_faces: Whether to save detected faces
        display_video: Whether to display the video feed
        confidence_threshold: Threshold for face detection confidence
        save_interval: Save faces every N frames
    """
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video opened: {frame_width}x{frame_height} at {fps} FPS")
    
    # Variables for tracking
    frame_count = 0
    faces_saved = 0
    last_faces = []
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame")
                break
                
            frame_count += 1
            
            # Process every other frame to improve performance
            if frame_count % 2 == 0:
                continue
                
            # Detect faces in frame
            faces, scores = multi_stage_face_detection(frame, confidence_threshold)
            
            # Create a copy of the frame for display
            if display_video:
                display_frame = frame.copy()
                
                # Draw rectangles around detected faces
                for (x, y, w, h), score in zip(faces, scores):
                    # Draw rectangle
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add confidence score text
                    label = f"Score: {score:.2f}"
                    cv2.putText(display_frame, label, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display number of faces detected
                face_count_text = f"Faces: {len(faces)} | Total saved: {faces_saved}"
                cv2.putText(display_frame, face_count_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow('Face Detection', display_frame)
            
            # Save faces at specified intervals
            if save_faces and frame_count % save_interval == 0 and len(faces) > 0:
                for (x, y, w, h), score in zip(faces, scores):
                    # Only save if the face is sufficiently different from previous faces
                    is_new_face = True
                    
                    # Simple check to avoid saving duplicate faces
                    for old_x, old_y, old_w, old_h in last_faces:
                        # Calculate center points
                        center_x, center_y = x + w//2, y + h//2
                        old_center_x, old_center_y = old_x + old_w//2, old_y + old_h//2
                        
                        # If centers are close and sizes are similar, consider it the same face
                        distance = np.sqrt((center_x - old_center_x)**2 + (center_y - old_center_y)**2)
                        size_ratio = (w*h) / (old_w*old_h) if old_w*old_h > 0 else 0
                        
                        if distance < (w + old_w) / 4 and 0.7 < size_ratio < 1.3:
                            is_new_face = False
                            break
                    
                    if is_new_face:
                        save_path = save_face_from_frame(frame, x, y, w, h, score, faces_dir, frame_count)
                        faces_saved += 1
                        print(f"Saved face: {save_path}")
                        
                        # Add to last_faces, keeping only the last 10
                        last_faces.append((x, y, w, h))
                        if len(last_faces) > 10:
                            last_faces.pop(0)
            
            # Break if 'q' is pressed
            if display_video and cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        # Clean up
        cap.release()
        if display_video:
            cv2.destroyAllWindows()
        
        print(f"Video processing complete. Total faces saved: {faces_saved}")

def main():
    try:
        print("Starting face detection from video...")
        
        # You can specify a video file path or use 0 for webcam
        video_source = 0  # Webcam
        # video_source = "path/to/your/video.mp4"  # Video file
        
        process_video(
            video_source="ClassGumbal.mp4",
            save_faces=True,
            display_video=True,
            confidence_threshold=0.75,
            save_interval=30  # Save faces every 30 frames
        )
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()