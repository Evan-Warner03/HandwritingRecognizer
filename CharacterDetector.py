import cv2
import os


class CharacterDetector(object):
    """CharacterDetector segments an image of writing

    into individual characters for recognition. It also detects

    spaces and formatting.
    """

    def __init__(self, image):
        # load the image if it is a path
        if isinstance(image, str):
            image = cv2.imread(image)
        self.image = image

    
    def convert_to_inverse_binary(self, image):
        """Converts an image to inverse binary

        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
        return binary_image
    

    def get_image_line_dimensions(self, image, image_height, threshold=1, minimum_size=50, padding=30):
        """Finds the dimension and location of each line of text in an image
        
        """

        # start by getting the average of each row
        # the idea is that rows with writing will have higher averages since we converted the image to inverse binary
        row_averages = cv2.reduce(image, 1, cv2.REDUCE_AVG).reshape(-1)
        line_upper_bounds, line_lower_bounds = [], []
        for row_index in range(image_height - 1):
            # check if the row_index is an upper or lower bound for a line of text
            if row_averages[row_index] <= threshold and row_averages[row_index + 1] > threshold:
                line_upper_bounds.append(row_index)
            elif row_averages[row_index] > threshold and row_averages[row_index+1] <= threshold and len(line_upper_bounds) > len(line_lower_bounds):
                line_lower_bounds.append(row_index)

        # determine the final line dimensions
        text_lines = []
        for i in range(len(line_upper_bounds)):
            if abs(line_lower_bounds[i] - line_upper_bounds[i]) > minimum_size:
                text_lines.append([line_upper_bounds[i] - padding, line_lower_bounds[i] + padding])
        return text_lines
        

    def segment_characters(self, debugging=False) -> list:
        """Segments self.image into a list of characters, separated by spaces and newlines

        to separate words and lines as detected
        """
        # convert the image to inverse binary
        binary_image = self.convert_to_inverse_binary(self.image)
        binary_image_height, binary_image_width = binary_image.shape[:2]

        # split the image into individual lines of text
        text_lines = self.get_image_line_dimensions(binary_image, binary_image_height, minimum_size=20, padding=30)

        # add debugging lines for line segmentation
        segmented_img = self.image.copy()
        if debugging:
            for line_upper_bound, line_lower_bound in text_lines:
                cv2.line(segmented_img, (0, line_upper_bound), (binary_image_width, line_upper_bound), (0, 0, 255), 5)
                cv2.line(segmented_img, (0, line_lower_bound), (binary_image_width, line_lower_bound), (0, 0, 255), 5)
            
        # split each line of text into individual characters
        # we transpose the image and take advantage of our existing line splitter
        chars = []
        for line_upper_bound, line_lower_bound in text_lines:
            transposed_text_line = cv2.transpose(binary_image[line_upper_bound:line_lower_bound])
            char_lines = self.get_image_line_dimensions(transposed_text_line, binary_image_width, threshold=2, minimum_size=8, padding=2)

            # add debugging lines for character segmentation
            if debugging:
                for char_upper_bound, char_lower_bound in char_lines:
                    cv2.line(segmented_img, (char_upper_bound, line_upper_bound), (char_upper_bound, line_lower_bound), (255, 0, 0), 5)
                    cv2.line(segmented_img, (char_lower_bound, line_upper_bound), (char_lower_bound, line_lower_bound), (0, 0, 255), 5)

            # save individual characters into array, separated by spaces between words and newlines between lines
            # find average character width, which will be used as a benchmark to determine when a word ends
            if len(char_lines):
                avg_char_width = sum([lower-upper for upper, lower in char_lines])/len(char_lines)
                prev = char_lines[0][0]
            for char_upper_bound, char_lower_bound in char_lines:
                # if there is a space between characters three or more times the width of the characters
                # this is a new word, so we should add a space
                if char_upper_bound - prev > avg_char_width:
                    chars.append(" ")
                # add the cropped character
                chars.append(self.image[line_upper_bound:line_lower_bound, char_upper_bound:char_lower_bound])
                prev = char_lower_bound
            # add a newline between each line
            chars.append("\n")
        
        # save the debugging image
        if debugging:
            cv2.imwrite("segmented.jpg", segmented_img)

        # return the segmented characters
        return chars