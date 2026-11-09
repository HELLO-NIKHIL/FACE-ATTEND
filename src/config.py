import yaml, os

def load_config(path:str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # expand relative paths relative to project root
    # assume this file resides in src/
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # fix paths
    tpath = cfg["storage"]["template_path"]
    if not os.path.isabs(tpath):
        cfg["storage"]["template_path"] = os.path.join(root, tpath)
    ldir = cfg["logging"]["dir"]
    if not os.path.isabs(ldir):
        cfg["logging"]["dir"] = os.path.join(root, ldir)
    return cfg
