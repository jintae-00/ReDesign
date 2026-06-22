import os
from omegaconf import OmegaConf

## yamlから読み込み
EXP_PARAMETERS = OmegaConf.load(os.path.abspath(__file__)[:-2] + "yaml")

## CONFIGの構成
HISAM_CONF = EXP_PARAMETERS["HISAM_CONF_default1"]
OCR_CONF = EXP_PARAMETERS["OCR_CONF_default1"]
LAMA_CONF = EXP_PARAMETERS["LAMA_CONF_default1"]
LAYOUT_CONF = EXP_PARAMETERS["LAYOUT_CONF_default1"]