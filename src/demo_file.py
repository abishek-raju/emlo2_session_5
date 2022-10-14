from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig

from src import utils

import numpy as np

from torchvision import transforms

log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    log.info(f"Loaded Model: {model}")
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def recognize_digit(image):
        if image is None:
            return None
        print(type(image))
        print(image.size)
        # image = torch.tensor(image, dtype=torch.float32)
        # image = transforms.ToTensor()(image).unsqueeze(0)
        image = np.moveaxis(image, -1, 0)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        print(image.shape)
        preds = model.forward_jit(image)
        preds = preds[0].tolist()
        return {labels[i]: preds[i] for i in range(10)}

    # im = gr.Image(shape=(28, 28), image_mode="L", invert_colors=True, source="canvas")
    im = gr.Image(type="numpy")

    demo = gr.Interface(
        fn=recognize_digit,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        live=True
    )

    demo.launch(server_name="0.0.0.0",server_port=8080)

@hydra.main(
    version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()