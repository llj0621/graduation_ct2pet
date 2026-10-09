import torch
from monai.transforms import MapTransform

__all__ = (
    "CapScanValued",
    "NormalizeScanValued",
    "cap_scan_value",
    "normalize_ct_scan",
    "cap_pet_scan_value",
    "normalize_pet_scan",
    "CapPETScanValued",
    "NormalizePETScanValued",
)


def cap_scan_value(scan: torch.Tensor) -> torch.Tensor:
    """
    Cap the values of a CT scan tensor to the range [-1000, 1000].

    Args:
        scan: The CT scan tensor.

    Returns:
        The capped CT scan tensor.
    """
    return torch.clamp(scan, min=-1000, max=1000)


def normalize_ct_scan(scan: torch.Tensor, to_one_one: bool = False) -> torch.Tensor:
    """
    Normalize the CT scan tensor.
    By default, scales [-1000, 1000] to [0, 1].
    If to_one_one is True, scales to [-1, 1].

    Args:
        scan: The CT scan tensor.
        to_one_one: Whether to normalize to [-1, 1] range.

    Returns:
        The normalized CT scan tensor.
    """
    # First cap it to be sure
    capped = cap_scan_value(scan)
    normalized = (capped + 1000) / 2000
    if to_one_one:
        normalized = normalized * 2.0 - 1.0
    return normalized


def cap_pet_scan_value(scan: torch.Tensor) -> torch.Tensor:
    """
    Cap the values of a PET scan tensor to the range [0, 5].

    Args:
        scan: The PET scan tensor.

    Returns:
        The capped PET scan tensor.
    """
    return torch.clamp(scan, min=0.0, max=5.0)


def normalize_pet_scan(scan: torch.Tensor, to_one_one: bool = False) -> torch.Tensor:
    """
    Normalize the PET scan tensor.
    By default, scales [0, 5] to [0, 1].
    If to_one_one is True, scales to [-1, 1].

    Args:
        scan: The PET scan tensor.
        to_one_one: Whether to normalize to [-1, 1] range.

    Returns:
        The normalized PET scan tensor.
    """
    # First cap it to be sure
    capped = cap_pet_scan_value(scan)
    normalized = capped / 5.0
    if to_one_one:
        normalized = normalized * 2.0 - 1.0
    return normalized


class CapScanValued(MapTransform):
    """
    Dictionary-based transform to cap CT scan values to [-1000, 1000].
    """

    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = cap_scan_value(d[key])
        return d


class NormalizeScanValued(MapTransform):
    """
    Dictionary-based transform to normalize CT scan values.
    """

    def __init__(
        self, keys, to_one_one: bool = False, allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)
        self.to_one_one = to_one_one

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = normalize_ct_scan(d[key], to_one_one=self.to_one_one)
        return d


class CapPETScanValued(MapTransform):
    """
    Dictionary-based transform to cap PET scan values to [0, 5].
    """

    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = cap_pet_scan_value(d[key])
        return d


class NormalizePETScanValued(MapTransform):
    """
    Dictionary-based transform to normalize PET scan values.
    """

    def __init__(
        self, keys, to_one_one: bool = False, allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)
        self.to_one_one = to_one_one

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = normalize_pet_scan(d[key], to_one_one=self.to_one_one)
        return d
