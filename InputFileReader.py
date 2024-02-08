


class InputFileReader(object):
    """
    InputFileReader is used to load a file from a filepath,
    and return the file data as an array of image data, split by page
    """

    def __init__(file_path: str):
        self.file_path = file_path
    

    def get_pages(self) -> list:
        # determine the file extension type
        pass