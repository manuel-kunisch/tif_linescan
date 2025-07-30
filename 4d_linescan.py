import numpy as np
import tifffile as tiff

def main(file_path):
    with tiff.TiffFile(file_path) as tif:
        # Read the first page of the TIFF file
        page = tif.pages[0]
        ij_metadata = tif.imagej_metadata

    image = tiff.imread(file_path)
    print(f"Image shape: {image.shape}")


if __name__ == "__main__":
    main('data/composite_zscan_green_ao_commandmatrix_red_pifoc_with_scalebar.tif')

