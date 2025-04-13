import os
import cv2
import argparse
from tqdm import tqdm

def prepare_image(input_path, output_path, size=256):
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        return False
    
    # Resize maintaining aspect ratio
    h, w = img.shape[:2]
    if h > w:
        new_h, new_w = size * h//w, size
    else:
        new_h, new_w = size, size * w//h
    img = cv2.resize(img, (new_w, new_h))
    
    # Center crop
    h, w = img.shape[:2]
    crop_y = (h - size) // 2
    crop_x = (w - size) // 2
    img = img[crop_y:crop_y+size, crop_x:crop_x+size]
    
    # Save image
    cv2.imwrite(output_path, img)
    return True

def process_directory(input_dir, output_dir, size=256):
    """Process all images in a directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image files
    valid_extensions = ['.jpg', '.jpeg', '.png']
    files = [f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in valid_extensions)]
    
    print(f"Processing {len(files)} images from {input_dir}")
    for file in tqdm(files):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        if not prepare_image(input_path, output_path, size):
            print(f"Failed to process {file}")

def main():
    parser = argparse.ArgumentParser(description='Prepare images for cartoon training')
    parser.add_argument('--input', required=True, help='Input directory containing images')
    parser.add_argument('--output', required=True, help='Output directory for processed images')
    parser.add_argument('--size', type=int, default=256, help='Output image size (default: 256)')
    args = parser.parse_args()
    
    process_directory(args.input, args.output, args.size)

if __name__ == '__main__':
    main()
