"""Reconstruct a CT NIfTI volume from one saved CT latent."""

from .decode_latent import run


def main() -> None:
    run("ct", "decode_ct.json")


if __name__ == "__main__":
    main()
