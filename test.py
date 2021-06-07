import face_recognition
from PIL import Image, ImageDraw
import numpy as np

print("choose photo number")
photo_number = input()

# load a sample pic
treka_image = face_recognition.load_image_file("./faces/known/treka.jpg")
treka_face_encoding = face_recognition.face_encodings(treka_image)[0]

# load another image

barakat_image = face_recognition.load_image_file("./faces/known/barakat.jpg")
barakat_face_encodings = face_recognition.face_encodings(barakat_image)[0]

# creat array of faces

known_face_encoding = [treka_face_encoding, barakat_face_encodings]
face_names = ["m.abo treka", "m.barakat"]

unknown_image = face_recognition.load_image_file(
    f"./faces/un_known/{photo_number}.jpg")

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image)

# convert image to PIL format

pil_image = Image.fromarray(unknown_image)

draw = ImageDraw.Draw(pil_image)

for (top, right, bottom, left), face_encodeig in zip(face_locations, face_encodings):

    matches = face_recognition.compare_faces(
        known_face_encoding, face_encodeig)

    name = "unkown"

    face_distance = face_recognition.face_distance(
        known_face_encoding, face_encodeig)

    best_match_index = np.argmin(face_distance)

    if matches[best_match_index]:
        name = face_names[best_match_index]

draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

text_width, text_height = draw.textsize(name)
draw.rectangle(((left, bottom - text_height - 10), (right, bottom)),
               fill=(0, 0, 255), outline=(0, 0, 255))
draw.text((left + 15, bottom - text_height - 5),
          name, fill=(255, 255, 255, 255))

del draw

pil_image.show()

new_image_name = input("name the new image :  ")

pil_image.save(new_image_name, "JPEG")
