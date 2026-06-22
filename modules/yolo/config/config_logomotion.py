import os
from omegaconf import OmegaConf

## yamlから読み込み
EXP_PARAMETERS = OmegaConf.load(os.path.abspath(__file__)[:-2] + "yaml")

## CONFIGの構成
Animation_CONF = EXP_PARAMETERS["Animation_CONF_default"]
Gemini_CONF = EXP_PARAMETERS["Gemini_CONF_default1"]
Render_CONF = EXP_PARAMETERS["Render_CONF_default1"]
