import cv2
import sys

from CharacterDetector import CharacterDetector
from CharacterRecognizer import CharacterRecognizer


def read_handwriting(image_path):
    """Returns the recognized handwriting from the image located at image_path

    """
    # segment image into characters
    detector = CharacterDetector(image_path)
    characters = detector.segment_characters(debugging=False) # enable debugging to see bounding box of characters

    # classify the individual characters
    recognizer = CharacterRecognizer()
    recognizer.load_hyperparameters("model_files/character_recognizer_hp.json")
    recognizer.load_model("model_files/character_recognizer_model.h5")
    recognizer.load_encodings("model_files/character_recognizer_encodings.json")

    # format the classified characters
    recognized_writing = ""
    for char in characters:
        # if the character is a space or newline, simply add it to the output
        if isinstance(char, str):
            recognized_writing += char
        else:
            # otherwise classify the character
            prediction, confidence = recognizer.classify_characters([char])[0]
            recognized_writing += prediction
    return recognized_writing.lower()


if __name__ == "__main__":
    handwriting = read_handwriting(sys.argv[1])
    with open("recognized_handwriting.txt", "w") as f:
        f.write(handwriting)