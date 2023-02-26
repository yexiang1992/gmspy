import gmspy as gm

GMdata = gm.loadPEERbatch(
    path="E:\_WorkSpace\JupyterWorkSpace\地震动强度参数研究\GroundMotions")
GMdata, Target_GMdata = gm.loadPEERbatch(
    path="E:\_WorkSpace\JupyterWorkSpace\地震动强度参数研究\GroundMotions", scale_base="PGA", scale_target=0.5)  # 0.5g
GMdata, Target_GMdata = gm.loadPEERbatch(
    path="E:\_WorkSpace\JupyterWorkSpace\地震动强度参数研究\GroundMotions", scale_base="Sa(1.0)", scale_target=0.5)  # 0.5g
