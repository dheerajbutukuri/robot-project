import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3 # pixels
FONT_THICKNESS = 2 
MODEL = "cnn"

#video = cv2.VideoCapture(0)

#main_dir = "D:\face recognition\"

# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color
    
print("loading known faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        print("file name:", filename)
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")

        #cv2.imshow(filename, image)
        cv2.waitKey(0)

        encoding = face_recognition.face_encodings(image)[0]
        
        known_faces.append(encoding)
        known_names.append(name)
       

print("loading UNknown faces")

for filename in os.listdir(UNKNOWN_FACES_DIR):
#while True:
    #print(f"{UNKNOWN_FACES_DIR}/{filename}")
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    cv2.imshow(filename, image)

    #print("processing video...")
    #ret, image = video.read()

    # resize the image for faster processing, windows is extremely slow
    #h,w,l = image.shape
    #resize_image = cv2.resize(image, (200, 150))

    locations = face_recognition.face_locations(resize_image, model=MODEL)
    encodings = face_recognition.face_encodings(resize_image, locations)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):

        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')

            # Each location contains positions in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
            color = name_to_color(match)

            # Paint frame
            cv2.rectangle(resize_image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels down
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Wite a name
            cv2.putText(resize_image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    cv2.imshow(filename, resize_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    #cv2.waitKey(0)
    #cv2.destroyWindow(filename)

video.release()
cv2.destroyAllWindows()

