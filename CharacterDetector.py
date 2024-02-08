import os
import cv2

class CharacterDetector(object):
    """
    CharacterDetector segments an image of writing
    into individual characters for recognition. It also detects
    spaces and formatting.
    """

    def __init__(self, image):
        self.image = image

    
    def convert_to_inverse_binary(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        return binary_image
    

    def get_image_line_dimensions(self, image, image_height, threshold=1, minimum_size=50, padding=30) -> list:
        # determines the dimensions of the lines of text of a given image

        # start by getting the average of each row
        # the idea is that rows with writing will have higher averages since we converted the image to inverse binary
        row_averages = cv2.reduce(image, 1, cv2.REDUCE_AVG).reshape(-1)
        line_upper_bounds, line_lower_bounds = [], []
        for row_index in range(image_height - 1):
            # check if the row_index is an upper or lower bound for a line of text
            if row_averages[row_index] <= threshold and row_averages[row_index + 1] > threshold:
                line_upper_bounds.append(row_index)
            elif row_averages[row_index] > threshold and row_averages[row_index+1] <= threshold:
                line_lower_bounds.append(row_index)
        
        # determine the final line dimensions
        text_lines = []
        for i in range(len(line_upper_bounds)-1):
            if line_lower_bounds[i] - line_upper_bounds[i] > minimum_size:
                text_lines.append([line_upper_bounds[i] - padding, line_lower_bounds[i] + padding])
        
        return text_lines
        

    def segment_characters(self) -> list:
        # convert the image to inverse binary
        binary_image = self.convert_to_inverse_binary(self.image)
        binary_image_height, binary_image_width = binary_image.shape[:2]

        # split the image into individual lines of text
        text_lines = self.get_image_line_dimensions(binary_image, binary_image_height)
        
        # split each line of text into individual characters
        # we transpose the image and take advantage of our existing line splitter
        c = 0
        for line_upper_bound, line_lower_bound in text_lines:
            transposed_text_line = cv2.transpose(binary_image[line_upper_bound:line_lower_bound])
            char_lines = self.get_image_line_dimensions(transposed_text_line, binary_image_width, threshold=5, minimum_size=20, padding=0)
            for char_upper_bound, char_lower_bound in char_lines:
                cv2.imwrite(f"./line_crops/char_{c}.jpg", binary_image[line_upper_bound:line_lower_bound, char_upper_bound:char_lower_bound])
                c += 1
    
if __name__ == "__main__":
    test_img = cv2.imread("./Test Images/IMG_2360.jpg")
    cd = CharacterDetector(test_img)
    cd.segment_characters()