from json import JSONEncoder


class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class Defaults:
    def __init__(self):
        # Global parameters
        self.threshold = 0.6
        self.max_size = None
        self.return_face_data = False
        self.return_landmarks = False
        self.extract_embedding = True
        self.extract_ga = False
        self.detect_masks = False
        self.api_ver = 1


class Models:
    def __init__(self):
        self.backend_name = 'trt'
        self.device = 'cuda'
        self.det_name = "retinaface_r50_v1"
        self.rec_name = "arcface_r100_v1"
        self.ga_name = None
        self.mask_detector = None
        self.rec_batch_size = 64
        self.det_batch_size = 1
        self.fp16 = True
        self.triton_uri = None


class EnvConfigs:
    def __init__(self):
        self.log_level = 'INFO'
        self.port = 18080

        self.models = Models()
        self.defaults = Defaults()
