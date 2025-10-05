import sys


class StreamToFile:
    def __init__(self, filename):
        self.file = open(filename, "w", buffering=1)

    def write(self, data):
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


def stream_to_file(filename):
    output_stream = StreamToFile(filename)
    sys.stdout = output_stream
    sys.stderr = output_stream
