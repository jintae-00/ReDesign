import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .hi_sam.modeling.build import model_registry
from .hi_sam.modeling.predictor import SamPredictor

def patchify(image: np.array, patch_size: int = 256):
    h, w = image.shape[:2]
    patch_list = []
    h_num, w_num = h // patch_size, w // patch_size
    h_remain, w_remain = h % patch_size, w % patch_size
    row, col = h_num + int(h_remain > 0), w_num + int(w_remain > 0)
    h_slices = [[r * patch_size, (r + 1) * patch_size] for r in range(h_num)]
    if h_remain:
        h_slices = h_slices + [[h - h_remain, h]]
    h_slices = np.tile(h_slices, (1, col)).reshape(-1, 2).tolist()
    w_slices = [[i * patch_size, (i + 1) * patch_size] for i in range(w_num)]
    if w_remain:
        w_slices = w_slices + [[w - w_remain, w]]
    w_slices = w_slices * row
    assert len(w_slices) == len(h_slices)
    for idx in range(0, len(w_slices)):
        # from left to right, then from top to bottom
        patch_list.append(
            image[
                h_slices[idx][0] : h_slices[idx][1],
                w_slices[idx][0] : w_slices[idx][1],
                :,
            ]
        )
    return patch_list, row, col


def unpatchify(patches, row, col):
    # return np.array
    whole = [np.concatenate(patches[r * col : (r + 1) * col], axis=1) for r in range(row)]
    whole = np.concatenate(whole, axis=0)
    return whole


def patchify_sliding(image: np.array, patch_size: int = 512, stride: int = 256):
    h, w = image.shape[:2]
    patch_list = []
    h_slice_list = []
    w_slice_list = []
    for j in range(0, h, stride):
        start_h, end_h = j, j + patch_size
        if end_h > h:
            start_h = max(h - patch_size, 0)
            end_h = h
        for i in range(0, w, stride):
            start_w, end_w = i, i + patch_size
            if end_w > w:
                start_w = max(w - patch_size, 0)
                end_w = w
            h_slice = slice(start_h, end_h)
            h_slice_list.append(h_slice)
            w_slice = slice(start_w, end_w)
            w_slice_list.append(w_slice)
            patch_list.append(image[h_slice, w_slice])

    return patch_list, h_slice_list, w_slice_list


def unpatchify_sliding(patch_list, h_slice_list, w_slice_list, ori_size):
    assert len(ori_size) == 2  # (h, w)
    whole_logits = np.zeros(ori_size)
    assert len(patch_list) == len(h_slice_list)
    assert len(h_slice_list) == len(w_slice_list)
    for idx in range(len(patch_list)):
        h_slice = h_slice_list[idx]
        w_slice = w_slice_list[idx]
        whole_logits[h_slice, w_slice] += patch_list[idx]

    return whole_logits

class HiSamSegmentationService:
    def __init__(
        self,
        checkpoint_dir,
        model_path,
        model_type="vit_h",
    ):
        args = self.get_args(model_type, model_path, checkpoint_dir)
        self.hisam = model_registry.get(model_type)(args)
        self.hisam.to(args.device)

        self.predictor = SamPredictor(self.hisam)

    def execute(self, image: Image.Image, patch_mode=False):
        image = np.array(image)
        if patch_mode:
            ori_size = image.shape[:2]
            patch_list, h_slice_list, w_slice_list = patchify_sliding(image, 512, 384)  # sliding window config
            mask_512 = []
            for patch in tqdm(patch_list):
                self.predictor.set_image(patch)
                m, hr_m, score, hr_score = self.predictor.predict(multimask_output=False, return_logits=True)
                assert hr_m.shape[0] == 1  # high-res mask
                mask_512.append(hr_m[0])
            mask_512 = unpatchify_sliding(mask_512, h_slice_list, w_slice_list, ori_size)
            assert mask_512.shape[-2:] == ori_size
            mask = mask_512
            mask = mask > self.predictor.model.mask_threshold
        else:
            self.predictor.set_image(image)
            mask, *_ = self.predictor.predict(multimask_output=False, hier_det=False, return_logits=True)
            mask = mask[0]
        print(mask.shape)
        mask = Image.fromarray(mask.astype(np.uint8) * 255)
        bin_mask = cv2.threshold(np.array(mask), 0, 255, 1)[1]
        return Image.fromarray(bin_mask), mask

    def get_args(self, model_type, model_path, checkpoint_dir):
        class Args:
            device = "cuda"
            input_size = [1024, 1024]
            patch_mode = False
            attn_layers = 1
            prompt_len = 12
            hier_det = False

        args = Args()
        args.checkpoint = model_path
        args.model_type = model_type
        args.checkpoint_dir = checkpoint_dir
        return args

class HiSam_Inference:
    def __init__(self, check_point_dir, model_path, model_type="vit_h"):
        self.hisam_service = HiSamSegmentationService(
            checkpoint_dir=check_point_dir,
            model_path=os.path.join(check_point_dir, model_path),
            model_type=model_type,
        )

    def __call__(self, input_img:Image.Image):
        bin_mask, mask = self.hisam_service.execute(input_img, patch_mode=False)
        return bin_mask, mask
