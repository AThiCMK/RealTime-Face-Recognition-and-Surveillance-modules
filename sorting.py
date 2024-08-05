import os
from re import I
import firebase_admin
from firebase_admin import credentials, storage
import cv2
import face_recognition
import shutil
import pickle

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r"D:\lpr\face_app\sorting-9ba6d-firebase-adminsdk-zlki1-e82684a056.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'sorting-9ba6d.appspot.com'})

class FaceRecognitionApp:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.root_folder = os.path.join(self.BASE_DIR, "ImagesAttendance")
        self.face_data_file = "face_data.pkl"
        self.all_face_encodings = []
        self.all_face_names = []
        self.load_face_data()
        self.local_folder_path = "D:/lpr/face_app/images/artha"

    def update_face_data(self, class_name=None):
        existing_face_encodings, existing_face_names, flag = self.load_face_data()

        if flag == 0:
            # Encode faces from the updated dataset
            new_face_encodings = []
            new_face_names = []

            if class_name:
                # Use the encoding function for a specific class
                average_face_encoding, class_name = self.encode_faces_in_class(os.path.join(self.root_folder, class_name))
                if average_face_encoding is not None:
                    new_face_encodings.append(average_face_encoding)
                    new_face_names.append(class_name)
            else:
                # Iterate through each folder in the root folder
                for class_name in os.listdir(self.root_folder):
                    class_folder = os.path.join(self.root_folder, class_name)

                    if os.path.isdir(class_folder):
                        # Check if the class is already present in the existing data
                        if class_name not in existing_face_names:
                            # Use the encoding function for a specific class
                            average_face_encoding, class_name = self.encode_faces_in_class(class_folder)
                            if average_face_encoding is not None:
                                new_face_encodings.append(average_face_encoding)
                                new_face_names.append(class_name)

            # Combine existing and new face data
            existing_face_encodings = existing_face_encodings + new_face_encodings
            existing_face_names = existing_face_names + new_face_names

        # Save the combined face data to the file using the existing save_face_data() function
        self.save_face_data(existing_face_encodings, existing_face_names)

        # Update class variables
        self.all_face_encodings = existing_face_encodings
        self.all_face_names = existing_face_names

    def load_face_data(self):
        # Check if the face data file exists
        if os.path.exists(self.face_data_file):
            # Load existing face data
            with open(self.face_data_file, 'rb') as file:
                face_data = pickle.load(file)
                existing_face_encodings = face_data.get('encodings', [])
                existing_face_names = face_data.get('names', [])
                flag = 0
        else:
            # Encode faces from the entire dataset
            existing_face_encodings, existing_face_names = self.encode_faces_in_dataset(self.root_folder)
            # Save the encoded data to the file using the existing save_face_data() function
            self.save_face_data(existing_face_encodings, existing_face_names)
            flag = 1
        self.all_face_encodings = existing_face_encodings
        self.all_face_names = existing_face_names
        return existing_face_encodings, existing_face_names, flag

    def save_face_data(self, encodings, names):
        face_data = {'encodings': encodings, 'names': names}
        with open(self.face_data_file, 'wb') as file:
            pickle.dump(face_data, file)
        print("Encodings Saved.")

    def encode_faces_in_class(self, class_folder):
        # Initialize face encodings and face names for the current class
        class_face_encodings = []
        class_face_names = []

        # Iterate through each image in the class folder
        for filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, filename)

            # Load the image
            image = face_recognition.load_image_file(image_path)

            # Find face encodings
            face_encodings = face_recognition.face_encodings(image)

            for face_encoding in face_encodings:
                # Use all face encodings (not just the first one)
                class_face_encodings.append(face_encoding)
                class_face_names.append(os.path.basename(class_folder))

        # Calculate the average face encoding for the class
        if class_face_encodings:
            average_face_encoding = [
                sum(emb[i] for emb in class_face_encodings) / len(class_face_encodings)
                for i in range(len(class_face_encodings[0]))
            ]
            return average_face_encoding, os.path.basename(class_folder)
        else:
            print("Class encoding failed")
            return None, None

    def encode_faces_in_dataset(self, root_folder):
        all_face_encodings = []
        all_face_names = []

        # Iterate through each folder in the root folder
        for class_name in os.listdir(root_folder):
            class_folder = os.path.join(root_folder, class_name)

            if os.path.isdir(class_folder):
                # Initialize face encodings and face names for the current class
                class_face_encodings = []
                class_face_names = []

                # Iterate through each image in the class folder
                for filename in os.listdir(class_folder):
                    image_path = os.path.join(class_folder, filename)

                    # Load the image
                    image = face_recognition.load_image_file(image_path)

                    # Find face encodings
                    face_encodings = face_recognition.face_encodings(image)

                    for face_encoding in face_encodings:
                        # Use all face encodings (not just the first one)
                        class_face_encodings.append(face_encoding)
                        class_face_names.append(class_name)

                # Calculate the average face encoding for the class
                if class_face_encodings:
                    average_face_encoding = [
                        sum(emb[i] for emb in class_face_encodings) / len(class_face_encodings)
                        for i in range(len(class_face_encodings[0]))
                    ]

                    # Add the average face encoding and name for the current class to the overall list
                    all_face_encodings.append(average_face_encoding)
                    all_face_names.append(class_name)

        return all_face_encodings, all_face_names

    def fetch_class_images(self, class_name):
    # Fetch class images from Firebase Storage to local folder
        bucket = storage.bucket()
        class_folder = os.path.join(self.local_folder_path, os.path.basename(class_name))
        os.makedirs(class_folder, exist_ok=True)
        blobs = bucket.list_blobs(prefix=class_name.replace("images/", "") + "/")
        for blob in blobs:
            relative_path = os.path.relpath(blob.name, class_name)
            local_filename = os.path.join(class_folder, relative_path)
            os.makedirs(os.path.dirname(local_filename), exist_ok=True)
            blob.download_to_filename(local_filename)
            print(f"Downloaded {blob.name} to {local_filename}")





    def sort_images(self, class_name, all_images_folder_path):
    # Sort images of a particular class from all images folder
        output_folder_path = os.path.join(self.local_folder_path, f"Sorted_{class_name}_Images")
        os.makedirs(output_folder_path, exist_ok=True)
        class_index = self.all_face_names.index(class_name)
        class_face_encoding = self.all_face_encodings[class_index]
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix="photos/")
        for blob in blobs:
            if blob.name.endswith(('.jpg', '.jpeg', '.png')):
            # Download image from Firebase Storage
                _, temp_local_filename = tempfile.mkstemp()
                blob.download_to_filename(temp_local_filename)
            # Load the image
                image = face_recognition.load_image_file(temp_local_filename)
            # Find face encodings
                face_locations = face_recognition.face_locations(image)
                if face_locations:
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    for face_encoding in face_encodings:
                        match = face_recognition.compare_faces([class_face_encoding], face_encoding, tolerance=0.5)
                        if match[0]:
                        # If the face matches the class, copy the image to the output folder
                            shutil.copy(temp_local_filename, output_folder_path)
                            print(f"Copied {blob.name} to {output_folder_path}")
                            break
        print(f"Images for class {class_name} have been sorted.")



    def recognize_faces(self, image_path):
        # Load image and recognize faces
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        names = []
        for face_encoding in face_encodings:
            # Compare face encoding with known encodings
            matches = face_recognition.compare_faces(self.all_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.all_face_names[first_match_index]
            names.append(name)
        return names

    def upload_image_to_firebase(self, image_path, class_name):
        # Upload sorted images back to Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob(f"images/{class_name}/{os.path.basename(image_path)}")
        with open(image_path, "rb") as image_file:
            blob.upload_from_file(image_file)
        print(f"Uploaded {os.path.basename(image_path)} to images/{class_name}")


if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.fetch_class_images("images/artha")
    app.sort_images("artha", "photos")

