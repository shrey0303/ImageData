import tensorflow as tf
import cv2
import itertools

mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()


class DataProcessing:
    def __init__(self, k: int, l: int, number: int, size: int):
        self.k = k
        self.l = l
        self.number = number
        self.size = size

    def resize_data(self, data) -> list:
        return list(map(lambda el: cv2.resize(el, (self.size, self.size)), data))

    def transform_data(self, data) -> list:
        return list(map(lambda el: list(itertools.chain(*el)), data))

    def generate_data(self, display: bool = False) -> list:
        indices = [
            i
            for i in range(len(x_train))
            if y_train[i] == self.k or y_train[i] == self.l
        ]
        input_data = [x_train[i] for i in indices][: self.number]
        output_data = [y_train[i] for i in indices][: self.number]
        input_data = self.resize_data(input_data)
        if not display:
            input_data = self.transform_data(input_data)
        data = [input_data, output_data]
        return data
