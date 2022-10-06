#!/usr/bin/env python3
#
import sys
import re

import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


class Detector:
    def __init__(self, labels, models):
        self.labels = load_labels(labels)
        self.interpreter = Interpreter(models)
        self.interpreter.allocate_tensors()
        _, h, w, _ = self.interpreter.get_input_details()[0]['shape']
        self.input_height = h
        self.input_width = w

    def detect(self, fileobj):
        image = Image.open(fileobj).convert('RGB').resize(
            (self.input_width, self.input_height), Image.Resampling.LANCZOS)
        return detect_objects(self.interpreter, image, 0.5)


def main():
    d = Detector("./model/coco_labels.txt", "./model/detect.tflite")
    r = d.detect(sys.stdin.buffer)
    for rr in r:
        print('id: %d, type: %s, score: %.2f' %
            (rr['class_id'], d.labels[rr['class_id']], rr['score']))


if __name__ == '__main__':
    main()

