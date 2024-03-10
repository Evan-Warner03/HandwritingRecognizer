# HandwritingRecognizer
A program that allows you to convert images of handwriting into plain text. Very useful for quickly digitizing handwritten lecture notes or assignment solutions. Recommended to be used in tandem with spellchecker to fix small recognition mistakes.

<br>

## Quick Start
1. Install the necessary dependencies
```
pip install requirements.txt
```

2. Recognize handwriting on any image!
```
python HandwritingReader.py path/to/your/image.py
```

<br>

## Example
```
python HandwritingReader.py test_images/IMG_4708.jpg
```
### Input Image
![Handwritten Assignment Answer](https://github.com/Evan-Warner03/HandwritingRecognizer/blob/c5437fcb063a170e8aa6b9d58171d80e7db87a70/test_images/IMG_4708.jpg?raw=true)

### Character Detection Bounding Boxes
![Detected Character Bounding Boxes](https://github.com/Evan-Warner03/HandwritingRecognizer/blob/c5437fcb063a170e8aa6b9d58171d80e7db87a70/source_images/segmented.jpg?raw=true)

### Plain Text Output
![Plain Text Output](https://github.com/Evan-Warner03/HandwritingRecognizer/blob/c5437fcb063a170e8aa6b9d58171d80e7db87a70/source_images/text.jpg?raw=true)  
(Perfect formatting and over 80% character accuracy even with messy handwriting!)

<br>

## Training Your Own CharacterRecognizer
You can use any dataset to train the CharacterRecognizer, provided it is structured as expected (see the file structure of character_dataset/):
```
recognizer = CharacterRecognizer()
recognizer.load_dataset("character_dataset")
```

CharacterRecognizer has a built in optimization method, which you can use to find the optimal hyperparameters for your model:
```
optimal = recognizer.compute_optimal_hyperparameters()
```

You can then train and save your model using the optimized hyperparameters:
```
# build model with optimal hyperparameters
recognizer.test_hyperparameters(
    num_conv_layers=5,
    min_conv_size=128,
    conv_increasing=1,
    num_dense_layers=1,
    min_dense_size=512,
    dense_increasing=1,
    num_epochs=25,
    batch_size=64
)

# save model and hyperparameters
recognizer.save_model()
recognizer.save_hyperparameters()
recognizer.save_encodings()
```

Finally you can load your newly trained CharacterRecognizer:
```
recognizer = CharacterRecognizer()
recognizer.load_hyperparameters("model_files/character_recognizer_hp.json")
recognizer.load_model("model_files/character_recognizer_model.h5")
recognizer.load_encodings("model_files/character_recognizer_encodings.json")
```
