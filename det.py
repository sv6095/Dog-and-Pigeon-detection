import cv2
import numpy as np
import os
import time
import torch
import requests  # For HTTP requests
from collections import defaultdict

from ultralytics import YOLO
import supervision as sv

MODEL_SIZE = "m" 
CONFIDENCE_THRESHOLD = 0.45  
MOVEMENT_THRESHOLD = 5  
APPROACHING_THRESHOLD = 0.05 

DOG_CLASS = 'dog'  
BIRD_CLASS = 'pigeon' 
TARGET_CLASSES = [DOG_CLASS, BIRD_CLASS]


# Use a configurable IP address (default to localhost)
ESP32_IP = os.environ.get("ESP32_IP", "http://192.168.1.100")

# Using a custom fine-tuned YOLOv8 model trained specifically on dog dataset
# Use relative path for better portability across systems
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolov8m_dog_finetuned.pt")
model = YOLO(model_path)
print(f"Model loaded successfully. Available classes: {list(model.names.values())}")


tracker = sv.ByteTrack()


object_history = defaultdict(list)
last_dispense_time = 0

dispensed_dogs = set()

# Add new constants
FEATURE_MATCH_THRESHOLD = 0.85
DOG_MEMORY_TIME = 300  
MIN_FEATURE_FRAMES = 5  


MIN_DISPENSE_INTERVAL = 10  


unique_dogs = {}  
dog_features = {}  


def get_box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def get_box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def is_approaching(track_id, current_box):
    if track_id not in object_history or len(object_history[track_id]) < 3:
        return False
    

    history = object_history[track_id]
    

    current_area = get_box_area(current_box)
    prev_areas = [get_box_area(box) for box in history[-3:]]
    avg_prev_area = sum(prev_areas) / len(prev_areas)
    

    growth_rate = (current_area - avg_prev_area) / max(avg_prev_area, 1)
    

    prev_center = get_box_center(history[-1])
    current_center = get_box_center(current_box)
    

    return growth_rate > APPROACHING_THRESHOLD


def is_moving(track_id, current_box):
    if track_id not in object_history or len(object_history[track_id]) < 2:
        return False

    history = object_history[track_id][-3:]
    current_center = get_box_center(current_box)
    

    for prev_box in history:
        prev_center = get_box_center(prev_box)
        distance = np.sqrt((current_center[0] - prev_center[0])**2 + 
                         (current_center[1] - prev_center[1])**2)
        if distance > MOVEMENT_THRESHOLD:
            return True
    
    return False

def log_event(event_type, animal_type, confidence, status):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {event_type}: {animal_type} ({confidence:.2f}) - {status}")


def dispense_food(animal_type, track_id, current_box, frame):
    global last_dispense_time
    current_time = time.time()
    
    if current_time - last_dispense_time < MIN_DISPENSE_INTERVAL:
        return False
    
    if animal_type == "dog":

        matched_dog = None
        for dog_id, dog_data in unique_dogs.items():
            if dog_data['track_id'] == track_id:
                matched_dog = dog_id
                break
        

        if not matched_dog:
            return False
            
        # Don't dispense if this unique dog has already been fed
        if matched_dog in dispensed_dogs:
            print(f"Dog {matched_dog} already fed, skipping dispensing")
            return False
        
        # Only dispense if approaching
        if is_approaching(track_id, current_box):
            print(f"\n----- FOOD DISPENSED FOR {animal_type.upper()} (Unique ID: {matched_dog}) -----\n")
            log_event("DISPENSE", animal_type, 1.0, f"Food dispensed for Unique ID: {matched_dog}")
            dispensed_dogs.add(matched_dog)  # Store unique ID instead of track ID
            last_dispense_time = current_time
            try:
                response = requests.get(f"{ESP32_IP}/dispense", timeout=2)
                if response.status_code == 200:
                    print(f"ESP32 Response: {response.text}")
                else:
                    print(f"ESP32 Error: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to connect to ESP32: {e}")
            return True
    elif animal_type == "pigeon":

        if is_approaching(track_id, current_box):
            print(f"\n----- FOOD DISPENSED FOR {animal_type.upper()} -----\n")
            log_event("DISPENSE", animal_type, 1.0, "Food dispensed")
            last_dispense_time = current_time
            return True
    
    return False

def extract_dog_features(frame, box):
    """Extract robust features from dog ROI"""
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate color histograms
    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    
    # Normalize histograms
    cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
    
    # Compute aspect ratio and relative size
    aspect_ratio = (x2 - x1) / (y2 - y1)
    
    return {
        'hue_hist': h_hist,
        'sat_hist': s_hist,
        'aspect_ratio': aspect_ratio,
        'last_seen': time.time()
    }

def match_dog_features(features1, features2):
    """Compare two sets of dog features"""
    h_score = cv2.compareHist(features1['hue_hist'], features2['hue_hist'], cv2.HISTCMP_CORREL)
    s_score = cv2.compareHist(features1['sat_hist'], features2['sat_hist'], cv2.HISTCMP_CORREL)
    
    # Compare aspect ratios
    ar_diff = abs(features1['aspect_ratio'] - features2['aspect_ratio'])
    ar_score = 1.0 - min(ar_diff / 0.5, 1.0)  # Normalize difference
    
    # Weighted combination
    return 0.4 * h_score + 0.4 * s_score + 0.2 * ar_score

def find_matching_dog(new_features):
    """Find if this dog matches any known dog"""
    current_time = time.time()
    best_match = None
    best_score = 0
    
    # Check against all known dogs
    for dog_id, dog_data in list(unique_dogs.items()):
        # Remove old dogs
        if current_time - dog_data['features']['last_seen'] > DOG_MEMORY_TIME:
            del unique_dogs[dog_id]
            continue
            
        score = match_dog_features(new_features, dog_data['features'])
        if score > best_score and score >= FEATURE_MATCH_THRESHOLD:
            best_score = score
            best_match = dog_id
    
    return best_match

def process_frame(frame):

    display_frame = frame.copy()
    

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    results = model.track(frame_rgb, persist=True, verbose=False)
    

    detected_dogs = False
    detected_birds = False
    approaching_dogs = False
    approaching_birds = False
    dispensed = False
    

    approaching_dog_list = []
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidence = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else None
        

        class_names = [model.names[class_id] for class_id in class_ids]
        

        current_dog_id = None
        current_dog_box = None
        
        # Process each detection
        for i in range(len(boxes)):
            if confidence[i] > CONFIDENCE_THRESHOLD:
                class_name = class_names[i]
                box = boxes[i]
                
                # Skip if not a target animal
                if class_name not in TARGET_CLASSES:
                    continue
                
                # Identify the animal type
                is_dog = (class_name == DOG_CLASS)
                is_bird = (class_name == BIRD_CLASS)
                
                # Update detection flags
                if is_dog:
                    detected_dogs = True
                    animal_type = "dog"
                elif is_bird:
                    detected_birds = True
                    animal_type = "bird (pigeon)"
                else:
                    continue
                
                # If we have tracking IDs available
                if track_ids is not None:
                    track_id = track_ids[i]
                    
                    # If this is a dog, store its ID for potential dispensing
                    if is_dog and track_id not in dispensed_dogs:
                        current_dog_id = track_id
                        current_dog_box = box
                    
                    # Store current box in history (keeping last 5 positions)
                    object_history[track_id].append(box)
                    if len(object_history[track_id]) > 5:
                        object_history[track_id].pop(0)
                    
                    # Compute growth rate if history available
                    if len(object_history[track_id]) >= 3:
                        current_area = get_box_area(box)
                        prev_areas = [get_box_area(b) for b in object_history[track_id][-3:]]
                        avg_prev_area = sum(prev_areas) / len(prev_areas)
                        growth_rate = (current_area - avg_prev_area) / max(avg_prev_area, 1)
                    else:
                        growth_rate = 0
                    
                    # Check if animal is moving or approaching
                    moving = is_moving(track_id, box)
                    
                    # Use computed growth_rate for status
                    if growth_rate > APPROACHING_THRESHOLD:
                        status = "APPROACHING"
                        # Record only unique track_id
                        if is_dog and track_id not in dispensed_dogs and track_id not in [tid for tid, _ in approaching_dog_list]:
                            approaching_dog_list.append((track_id, box))
                        if is_bird:
                            approaching_birds = True
                    elif moving:
                        status = "MOVING"
                    else:
                        status = "STATIONARY"
                    
                    # Color based on movement/approach status
                    color = (0, 0, 255) if status=="APPROACHING" else (0, 255, 255) if status=="MOVING" else (0, 255, 0)
                    
                    # Create label
                    label = f"{animal_type}-{track_id} ({confidence[i]:.2f}) {status}"
                    
                    # Occasionally log events for monitoring
                    if status=="APPROACHING" and np.random.random() < 0.1:  # Log ~10% of approaching events
                        log_event("TRACK", animal_type, confidence[i], status)
                
                else:
                    # Without tracking, just detect
                    color = (0, 255, 0)
                    label = f"{animal_type} ({confidence[i]:.2f})"
                
                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add filled background for text for better readability
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(display_frame, 
                              (x1, y1 - 20), 
                              (x1 + text_size[0], y1), 
                              color, 
                              -1)  # -1 means filled
                
                cv2.putText(display_frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # For dogs, build up feature history
                if is_dog:
                    if track_id not in dog_features:
                        dog_features[track_id] = []
                    
                    # Add current frame features
                    try:
                        current_features = extract_dog_features(frame, box)
                        dog_features[track_id].append(current_features)
                        
                        # Once we have enough frames, try to identify the dog
                        if len(dog_features[track_id]) >= MIN_FEATURE_FRAMES:
                            # Average the features
                            avg_features = {
                                'hue_hist': sum(f['hue_hist'] for f in dog_features[track_id]) / len(dog_features[track_id]),
                                'sat_hist': sum(f['sat_hist'] for f in dog_features[track_id]) / len(dog_features[track_id]),
                                'aspect_ratio': sum(f['aspect_ratio'] for f in dog_features[track_id]) / len(dog_features[track_id]),
                                'last_seen': current_features['last_seen']
                            }
                            
                            # Try to match with known dogs
                            matched_dog = find_matching_dog(avg_features)
                            if matched_dog:
                                # Update existing dog
                                unique_dogs[matched_dog]['features'] = avg_features
                                unique_dogs[matched_dog]['track_id'] = track_id
                            else:
                                # New unique dog
                                new_id = f"dog_{len(unique_dogs) + 1}"
                                unique_dogs[new_id] = {
                                    'features': avg_features,
                                    'track_id': track_id,
                                    'first_seen': time.time()
                                }
                    except Exception as e:
                        print(f"Error processing dog features: {e}")
    
    # Dispense food for approaching dogs
    dispensed = False
    for dog_id, dog_box in approaching_dog_list:
        # Only consider dogs we've identified
        if any(dog_data['track_id'] == dog_id for dog_data in unique_dogs.values()):
            if dispense_food("dog", dog_id, dog_box, frame):
                dispensed = True
                break
    
    # Handle birds as before
    if not dispensed and approaching_birds:
        dispensed = dispense_food("pigeon", track_id, box, frame)
    
    # Add count of approaching dogs to display
    if approaching_dogs:
        unfed_approaching = len([1 for dog_id, _ in approaching_dog_list 
                               if dog_id not in dispensed_dogs])
        if unfed_approaching > 0:
            print(f"\nNumber of approaching unfed dogs: {unfed_approaching}")
    
    return (
        display_frame,
        detected_dogs,
        detected_birds,
        approaching_dogs,
        approaching_birds,
        dispensed
    )

# Add info text to the frame
def add_info_text(frame, text, position, color=(255, 255, 255)):
    # Add background for text (better readability)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, 
                 (position[0]-5, position[1]-20), 
                 (position[0] + text_size[0]+5, position[1]+5), 
                 (0, 0, 0), 
                 -1)  # -1 means filled
    
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    print("\n=============================================")
    print("  DOG AND PIGEON DETECTION SYSTEM")
    print("=============================================\n")
    
    print(f"Configuration:")
    print(f"- Model: Custom YOLOv8{MODEL_SIZE} fine-tuned on dog dataset")
    print(f"- Target animals: Dogs and Pigeons/Birds")
    print(f"- Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"- Movement threshold: {MOVEMENT_THRESHOLD} pixels")
    print(f"- Approaching threshold: {APPROACHING_THRESHOLD*100}% increase in size")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Track statistics
    stats = {
        "food_dispensed": 0,
        "start_time": time.time()
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Process the frame
            processed_frame, detected_dogs, detected_birds, approaching_dogs, approaching_birds, dispensed = process_frame(frame)
            
            if dispensed:
                stats["food_dispensed"] += 1
            
            # Add header with date and time
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            add_info_text(processed_frame, f"Date/Time: {current_time}", (10, 30), (255, 255, 255))
            
            # Status info
            add_info_text(processed_frame, "Status:", (10, 70))
            
            # Display status based on detection results
            if dispensed:
                add_info_text(processed_frame, "FOOD DISPENSED", (120, 70), (0, 0, 255))
            elif approaching_dogs:
                add_info_text(processed_frame, "DOG APPROACHING", (120, 70), (0, 0, 255))
            elif approaching_birds:
                add_info_text(processed_frame, "PIGEON APPROACHING", (120, 70), (0, 0, 255))
            elif detected_dogs:
                add_info_text(processed_frame, "DOG DETECTED", (120, 70), (0, 255, 0))
            elif detected_birds:
                add_info_text(processed_frame, "PIGEON DETECTED", (120, 70), (0, 255, 0))
            else:
                add_info_text(processed_frame, "MONITORING", (120, 70), (255, 255, 255))

            
            # Show detection counts at the bottom of the frame
            runtime = int(time.time() - stats["start_time"])
            stats_text = f"Runtime: {runtime//3600:02d}h:{(runtime%3600)//60:02d}m:{runtime%60:02d}s | "
            stats_text += f"Food: {stats['food_dispensed']}"
            
            h, w = processed_frame.shape[:2]
            add_info_text(processed_frame, stats_text, (10, h-20), (200, 200, 200))
            
                
            # Display the result
            cv2.imshow("Dog & Pigeon Detection", processed_frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        runtime = int(time.time() - stats["start_time"])
        print("\n---------------------------------------------")
        print("System shutdown summary:")
        print(f"- Runtime: {runtime//3600:02d}:{(runtime%3600)//60:02d}:{runtime%60:02d}")
        print(f"- Food dispensed: {stats['food_dispensed']}")
        print("---------------------------------------------\n")

if __name__ == "__main__":
    main()

