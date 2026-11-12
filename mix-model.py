import math
from pathlib import Path
import numpy as np
import torch
from competition_api import Sample
from cnn_common import IN_CH, Net, build_input, meta_vector
from spec_common import SpecNet, optimize_net

HERE = Path(__file__).parent


class Model:
    def __init__(self):
        torch.set_num_threads(1)
        dummy_inp = torch.randn(1, IN_CH, 256, dtype=torch.float32)

        # 1. Cerebro Generalista
        ck_gen = torch.load(HERE / "cnn.pt", map_location="cpu")
        self.use_meta = ck_gen["use_meta"]
        net_gen = Net(ck_gen["width"])
        net_gen.load_state_dict(ck_gen["state"])
        self.net_gen = optimize_net(net_gen, dummy_inp)

        # 2. Cerebro Especialista 6 vs 7
        ck_spec = torch.load(HERE / "cnn_especialista.pt", map_location="cpu")
        net_spec = SpecNet(ck_spec["width"])
        net_spec.load_state_dict(ck_spec["state"])
        self.net_spec = optimize_net(net_spec, dummy_inp)

        self.alpha = float(ck_spec.get("alpha", 0.65))
        self.thr7 = float(ck_spec.get("thr7", 0.50))

        dummy = Sample(np.random.randn(4, 256).astype(np.float32), {"sample_rate": 1, "group": "A", "level": 0.0})
        for _ in range(30):
            self.predict(dummy)

    def predict(self, x: Sample) -> int:
        mv = meta_vector(x.meta) if self.use_meta else np.zeros(4, np.float32)
        inp = torch.from_numpy(build_input(x.series, mv)[None])

        with torch.inference_mode():
            out_gen = self.net_gen(inp)[0]
            pred_gen = int(out_gen.argmax())

            if pred_gen >= 6:
                diff_gen = float(out_gen[7] - out_gen[6])
                if abs(diff_gen) < 1.3:
                    out_spec = self.net_spec(inp)[0]
                    diff_spec = float(out_spec[1] - out_spec[0])
                    p7_gen = 1.0 / (1.0 + math.exp(-diff_gen))
                    p7_spec = 1.0 / (1.0 + math.exp(-diff_spec))
                    p7 = (1.0 - self.alpha) * p7_gen + self.alpha * p7_spec
                    return 7 if p7 >= self.thr7 else 6

            return pred_gen