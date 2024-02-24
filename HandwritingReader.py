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
    characters = detector.segment_characters()

    # classify the individual characters
    recognizer = CharacterRecognizer()
    recognizer.load_model("character_recognizer.h5")
    characters = recognizer.classify_characters(characters)

    print(characters)