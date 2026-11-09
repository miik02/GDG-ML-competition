from pathlib import Path

import numpy as np
import torch

from competition_api import Sample
from cnn_common import Net, build_input, meta_vector

HERE = Path(__file__).parent


class Model:
    def __init__(self):
        torch.set_num_threads(1)
        ck = torch.load(HERE / "cnn.pt", map_location="cpu")
        self.use_meta = ck["use_meta"]
        self.net = Net(ck["width"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        dummy = Sample(np.random.randn(4, 256).astype(np.float32), {"sample_rate": 1, "group": "A", "level": 0.0})
        for _ in range(20):  # precalentamiento fuera del reloj
            self.predict(dummy)

    def predict(self, x: Sample) -> int:
        mv = meta_vector(x.meta) if self.use_meta else np.zeros(4, np.float32)
        inp = torch.from_numpy(build_input(x.series, mv)[None])
        with torch.inference_mode():
            return int(self.net(inp).argmax(1))