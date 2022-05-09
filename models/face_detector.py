import cv2
import numpy as np

from .abstract_detector import AbstractDetector


class FaceDetector(AbstractDetector):
    def __init__(
            self, prototxt, weights, input_size, confidence_threshold,
            input_scale=1.0, input_swap_rb=False, use_gpu=True,
            extract_best=True
    ):
        super().__init__(
            weights=weights,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
            input_scale=input_scale,
            input_swap_rb=input_swap_rb,
            use_gpu=use_gpu
        )
        self.__prototxt = prototxt
        self.__extract_best = extract_best
        self._net = self._init_net()
        self.__run_fake_inference()

    def _init_net(self):
        net = cv2.dnn.readNetFromCaffe(self.__prototxt, self._weights)
        if self._use_gpu:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def __run_fake_inference(self):
        fake_shape = (1, self._input_size, self._input_size, 3)
        fake_net_input = np.random.randint(0, 256, fake_shape).astype('uint8')
        self._set_net_input(fake_net_input)
        _ = self._net.forward()

    def detect(self, batch):
        frames_dimensions = self.__get_frames_dimensions(batch)
        self._set_net_input(batch)
        net_output = self._net.forward()
        frames_indices, boxes, confs = self.__process_net_output(
            net_output, frames_dimensions
        )
        return frames_indices, boxes, confs

    @staticmethod
    def __get_frames_dimensions(batch):
        frames_dimensions = np.array([frame.shape[0:2] for frame in batch])
        frames_dimensions = np.repeat(frames_dimensions, 2, axis=1)
        frames_dimensions = frames_dimensions[:, [2, 1, 3, 0]]
        return frames_dimensions

    def __process_net_output(self, net_output, frames_dimensions):
        detections, frames_indices = self.__extract_detections_by_confidence(
            net_output
        )
        if len(detections) > 0:
            frames_indices, boxes, confs = self.__continue_processing(
                detections, frames_indices, frames_dimensions
            )
        else:
            frames_indices, boxes, confs = self.__stop_processing()
        return frames_indices, boxes, confs

    def __extract_detections_by_confidence(self, net_output):
        mask = net_output[0, 0, :, 2] > self._confidence_threshold
        detections = net_output[0, 0, mask, :]
        frames_indices = detections[:, 0].astype('int')
        return detections, frames_indices

    def __continue_processing(
            self, detections, frames_indices, frames_dimensions
    ):
        if self.__extract_best:
            extract_output = self.__extract_the_best_detection_per_frame(
                detections, frames_indices
            )
            detections, frames_indices = extract_output
        boxes = self.__get_boxes(
            detections, frames_indices, frames_dimensions
        )
        confs = self.__get_confidences(detections)
        return frames_indices, boxes, confs

    @staticmethod
    def __stop_processing():
        frames_indices = np.array([]).astype('int')
        boxes = np.zeros((0, 4)).astype('int')
        confs = np.array([]).astype('float')
        return frames_indices, boxes, confs

    @staticmethod
    def __extract_the_best_detection_per_frame(detections, frames_indices):
        unique_frames_indices = np.unique(frames_indices)
        unique_frames_indices = np.reshape(unique_frames_indices, (-1, 1))
        equality_matrix = unique_frames_indices == frames_indices
        equality_matrix = np.expand_dims(equality_matrix, axis=2)
        extended_detections = np.where(equality_matrix, detections, -np.inf)
        range_like_unique = np.arange(len(unique_frames_indices))
        mask = np.argmax(extended_detections[:, :, 2], axis=1)
        detections = extended_detections[range_like_unique, mask, :]
        frames_indices = detections[:, 0].astype('int')
        return detections, frames_indices

    @staticmethod
    def __get_boxes(detections, frames_indices, frames_dimensions):
        boxes = detections[:, 3:7]
        boxes *= frames_dimensions[frames_indices, :]
        boxes = boxes.astype('int')
        return boxes

    @staticmethod
    def __get_confidences(detections):
        confs = detections[:, 2]
        return confs
