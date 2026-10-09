"""Legacy adjacent-to-scan latent path compatibility."""

def construct_latent_path(path_to_scan_file: str) -> str:
    """
    Construct the path where CT Latent PT files is supposed to be based on the path of CT Scan file.
    If the file is xx/aa/bb.nii.gz or xx/aa/bb.nii, the resulted CT Latent should be stored as xx/aa/bb.pt
    """
    base_path = path_to_scan_file
    if base_path.endswith(".nii.gz"):
        base_path = base_path[:-7]
    elif base_path.endswith(".nii"):
        base_path = base_path[:-4]
    return base_path + ".pt"
