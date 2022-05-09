import abc
import cv2


NOT_IMPLEMENTED_ERROR_MESSAGE = 'Method is not implemented!'


class AbstractDetector:
    def __init__(
            self, weights, input_size, confidence_threshold,
            input_scale=1.0, input_swap_rb=True, use_gpu=True
    ):
        self._weights = weights
        self._input_size = input_size
        self._confidence_threshold = confidence_threshold
        self._input_scale = input_scale
        self._input_swap_rb = input_swap_rb
        self._use_gpu = use_gpu
        self._net = None

    @abc.abstractmethod
    def _init_net(self):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)

    @abc.abstractmethod
    def detect(self, batch):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)

    def _set_net_input(self, batch):
        blob_mean = (104.0, 177.0, 123.0)
        net_input = cv2.dnn.blobFromImages(
            batch, self._input_scale, (self._input_size, self._input_size),
            blob_mean, self._input_swap_rb
        )
        self._net.setInput(net_input)
