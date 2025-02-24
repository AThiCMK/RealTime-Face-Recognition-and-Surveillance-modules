
import os
import cv2
import face_recognition
import shutil

class SortImagesBackend:
    def __init__(self, face_encodings, face_names):
        self.all_face_encodings = face_encodings
        self.all_face_names = face_names

    def sort_images(self, folder_path, class_name):
        if class_name not in self.all_face_names:
            print(f"Class {class_name} not found in the dataset.")
            return

        class_index = self.all_face_names.index(class_name)
        class_face_encoding = self.all_face_encodings[class_index]
        output_folder_path = os.path.join(folder_path, f"Sorted_{class_name}_Images")
        os.makedirs(output_folder_path, exist_ok=True)

        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)

                for face_encoding in face_encodings:
                    match = face_recognition.compare_faces([class_face_encoding], face_encoding, tolerance=0.5)
                    if match[0]:
                        shutil.copy(image_path, os.path.join(output_folder_path, filename))
                        break

        print(f"Images for class {class_name} have been sorted.")