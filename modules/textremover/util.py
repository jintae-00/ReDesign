from PIL import Image, ImageOps
import cv2
import numpy as np

def crate_mask(mask:Image.Image, extend_mask_size=10):
    mask = mask.convert("L")
    binary = np.array(mask.point(lambda p: 255 if p >= 128 else 0))# 強制2値化
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extended_mask = np.zeros_like(binary)
    for contour in contours:
        cv2.drawContours(extended_mask, [contour], -1, 255, extend_mask_size) # マスクを拡張
        cv2.drawContours(extended_mask, [contour], -1, 255, -1) # マスクの中をすべて塗りつぶし

    return Image.fromarray(extended_mask)


class ResizeKeepAspectRatio:
    def __init__(self, original_image):
        self.original_image = original_image
        self.original_size = original_image.size

    def forward(self, original_image=None, target_size=None, bg_color=(255, 255, 255)):
        if original_image == None:
            original_image = self.original_image

        if target_size == None:
            target_size = self.original_size

        original_width, original_height = original_image.size
        target_width, target_height = target_size

        aspect_ratio = original_width / original_height
        if aspect_ratio > target_width / target_height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_width = int(target_height * aspect_ratio)
            new_height = target_height

        resized_image = original_image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

        new_image = Image.new("RGB", target_size, bg_color)
        paste_position = (
            (target_width - new_width) // 2,
            (target_height - new_height) // 2
        )
        new_image.paste(resized_image, paste_position)
        return new_image
    
    def reverse(self, resized_image:Image.Image):
        mask = Image.new("RGB", self.original_size, (0,0,0))
        mask = self.forward(original_image=mask, target_size=resized_image.size)
        mask = ImageOps.invert(mask).convert("L")

        resized_image = resized_image.crop(mask.getbbox())
        resized_image = resized_image.resize(self.original_size, resample=Image.Resampling.LANCZOS)
        return resized_image