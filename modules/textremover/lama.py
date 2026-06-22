import os
import torch
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageChops
from .util import crate_mask, ResizeKeepAspectRatio

# ── GPU 메모리 로그 헬퍼 ────────────────────────────────────────
def gpu_mem(tag: str):
    if not torch.cuda.is_available():
        print(f"[{tag}]  CUDA unavailable")
        return
    torch.cuda.synchronize()
    alloc  = torch.cuda.memory_allocated()  / 1024**2  # MB
    reserv = torch.cuda.memory_reserved()   / 1024**2
    print(f"[{tag}]  alloc={alloc:8.1f} MB   reserv={reserv:8.1f} MB")

def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img

class LaMa:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_jit_model(model_path).eval()

    def load_jit_model(self, model_path):
        model = torch.jit.load(model_path, map_location="cpu").to(self.device)
        model.eval()
        return model

    def forward(self, image, mask):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return Image.fromarray(cur_res)
    
    def remove_text_by_mask(
            self,
            base_image: Image.Image,
            mask      : Image.Image,
            max_square: int = 2048   # ↑ GPU 여유에 따라 1024, 4096 등으로 조정
        ):
        """
        - 원본 해상도를 그대로 유지(다운샘플 X)
        - 단, 너무 큰 이미지는 max_square 이하로 축소
        - LaMa는 8의 배수만 맞으면 되므로 padding 방식 우선
        - 기존 crate_mask / alpha 합성 로직 유지
        Returns
        -------
        image             : inpainted RGB Image (원본 해상도)
        masked_base_image : 알파가 포함된 inpainting 대상 미리보기
        mask              : crate_mask로 후처리된 L(Image)
        """

        # ── 0. 준비
        w0, h0 = base_image.size
        max_dim = max(w0, h0)

        # 1) square_size 결정
        if max_dim <= max_square:
            # 다운리사이즈 없이 padding만
            square_size = ((max_dim + 7) // 8) * 8  # 8의 배수
            resize_func = None
            base_proc   = ImageOps.expand(
                base_image,
                border=(0, 0, square_size - w0, square_size - h0),
                fill=(0, 0, 0))
            mask_proc   = ImageOps.expand(
                mask,
                border=(0, 0, square_size - w0, square_size - h0),
                fill=0)
        else:
            # GPU 메모리 보호용 다운리사이즈(Aspect-ratio 유지)
            square_size = max_square
            resize_func = ResizeKeepAspectRatio(base_image)
            base_proc   = resize_func.forward(target_size=(square_size, square_size))
            mask_proc   = resize_func.forward(mask, target_size=(square_size, square_size),
                                            bg_color=(0, 0, 0))

        # 2) 마스크 후처리 (기존 로직 유지)
        base_proc = base_proc.convert("RGB")
        mask_proc = crate_mask(mask_proc.convert("L"))

        _mask = ImageOps.invert(mask_proc)
        masked_base_image = base_proc.copy()
        masked_base_image.putalpha(_mask)                     # 시각화용

        # 3) LaMa 추론
        image_out = self.forward(
            image=np.asarray(base_proc)[:, :, ::-1],        # RGB 그대로
            mask=np.asarray(mask_proc)
        )

        # 4) 원본 해상도로 복원
        if resize_func is not None:
            image_out         = resize_func.reverse(image_out)
            masked_base_image = resize_func.reverse(masked_base_image)
            mask_proc         = resize_func.reverse(mask_proc)
        else:
            # padding만 했던 경우 → 패딩 영역 crop
            image_out         = image_out.crop((0, 0, w0, h0))
            masked_base_image = masked_base_image.crop((0, 0, w0, h0))
            mask_proc         = mask_proc.crop((0, 0, w0, h0))

        return image_out, masked_base_image, mask_proc
    
    # 이전 함수
    # def remove_text_by_mask(self, base_image:Image.Image, mask:Image.Image):
    #     # square_size =  1024 if np.max(base_image.size) > 512 else 512
    #     square_size = 512 #fix

    #     resize_func = ResizeKeepAspectRatio(base_image)
    #     base_image = resize_func.forward(target_size=(square_size, square_size))
    #     base_image = base_image.convert("RGB")
    #     mask = resize_func.forward(mask, target_size=(square_size, square_size), bg_color=(0,0,0)).convert("L")
    #     mask = crate_mask(mask)

    #     _mask = ImageOps.invert(mask)
    #     masked_base_image = ImageChops.multiply(base_image, Image.merge("RGB", (_mask, _mask, _mask)))
    #     masked_base_image = base_image.copy()
    #     masked_base_image.putalpha(_mask)

    #     image = self.forward(
    #         image=np.array(base_image)[:, :, ::-1],
    #         mask=np.array(mask),
    #     )
    #     image = resize_func.reverse(image)
    #     masked_base_image = resize_func.reverse(masked_base_image)
    #     mask = resize_func.reverse(mask)

    #     return image, masked_base_image, mask