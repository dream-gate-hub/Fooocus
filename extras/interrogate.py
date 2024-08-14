import os
import torch
import ldm_patched.modules.model_management as model_management

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from modules.model_loader import load_file_from_url
from modules.config import path_clip_vision
from ldm_patched.modules.model_patcher import ModelPatcher
from extras.BLIP.models.blip import blip_decoder


blip_image_eval_size = 384
blip_repo_root = os.path.join(os.path.dirname(__file__), 'BLIP')


class Interrogator:
    def __init__(self):
        self.blip_model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

    @torch.no_grad()
    @torch.inference_mode()
    def interrogate(self, img_rgb):
        if self.blip_model is None:
            filename = load_file_from_url(
                url='https://huggingface.co/lllyasviel/misc/resolve/main/model_base_caption_capfilt_large.pth',
                model_dir=path_clip_vision,
                file_name='model_base_caption_capfilt_large.pth',
            )

            model = blip_decoder(pretrained=filename, image_size=blip_image_eval_size, vit='base',
                                 med_config=os.path.join(blip_repo_root, "configs", "med_config.json"))
            model.eval()

            if model_management.should_use_fp16(device=self.device):
                model.half()
                self.dtype = torch.float16

            model.to(self.device)

            self.blip_model = ModelPatcher(model, load_device=self.device, offload_device=self.device)

        # model_management.load_model_gpu(self.blip_model)

        gpu_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(img_rgb).unsqueeze(0).to(self.device, dtype=self.dtype)

        caption = self.blip_model.model.generate(gpu_image, sample=True, num_beams=1, min_length=20, max_length=150)[0]

        return caption
        # return 0


default_interrogator = Interrogator().interrogate


