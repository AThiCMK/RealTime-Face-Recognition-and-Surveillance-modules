import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from PIL import Image
import os
import numpy as np

# Set up GPU environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained models
mtcnn = MTCNN(device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load class names and corresponding images
root_folder = "ImagesAttendance"
class_names = os.listdir(root_folder)

# Process images for each class
class_embeddings = {}
for class_name in class_names:
    class_folder = os.path.join(root_folder, class_name)
    image_paths = [os.path.join(class_folder, img) for img in os.listdir(class_folder)]
    embeddings = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = mtcnn(img)
        if img is not None:  # Check if mtcnn returns a non-None value
            embeddings.append(model(img.unsqueeze(0).to(device)).detach().cpu().numpy())
    class_embeddings[class_name] = embeddings

# Real-time video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None and len(boxes) > 0:  # Check if boxes is not empty
        boxes = [box.astype(int) for box in boxes]  # Convert box coordinates to integers
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            face_region = frame[y1:y2, x1:x2]  # Extract face region from frame

            # Convert face region to PIL Image
            face_img = Image.fromarray(face_region).convert('RGB')

            # Recognize the face
            with torch.no_grad():
                face_tensor = mtcnn(face_img).unsqueeze(0).to(device)
                if face_tensor is not None:
                    embedding = model(face_tensor).detach().cpu().numpy()

                    # Compare with known embeddings to recognize the person
                    recognized_class = "Unknown"
                    min_distance = float('inf')
                    for class_name, class_embeddings_list in class_embeddings.items():
                        for known_embedding in class_embeddings_list:
                            distance = np.linalg.norm(known_embedding - embedding)
                            if distance < min_distance:
                                min_distance = distance
                                recognized_class = class_name

                    # Display results
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, recognized_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()