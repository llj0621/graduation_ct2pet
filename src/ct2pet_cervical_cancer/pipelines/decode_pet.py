"""Reconstruct a PET NIfTI volume from one saved PET latent."""

from .decode_latent import run


def main() -> None:
    run("pet", "decode_pet.json")


if __name__ == "__main__":
    main()
