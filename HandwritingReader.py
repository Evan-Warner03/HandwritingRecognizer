import cv2
from CharacterDetector import CharacterDetector
from CharacterRecognizer import CharacterRecognizer

def read_handwriting(image_path):
    # reads the handwriting in the given image, and returns it as plain text
    detector = CharacterDetector()

if __name__ == "__main__":
    # load image
    test_img = cv2.imread("./Test Images/IMG_2360.jpg")

    # segment the image into individual characters
    detector = CharacterDetector(test_img)
    print("Loaded Character Detector")
    characters = detector.segment_characters()
    print("Segmented Characters")

    # classify the individual characters
    recognizer = CharacterRecognizer()
    recognizer.load_hyperparameters("character_recognizer_hp.json")
    recognizer.load_model("character_recognizer_model.h5")
    recognizer.load_encodings("character_recognizer_encodings.json")
    print("Loaded Character Recognizer")
    character_predictions = recognizer.classify_characters(characters)
    print("Classified characters")

    out = ""
    for pred, conf in character_predictions:
        out += pred
    print(out)