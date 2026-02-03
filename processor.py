#!/usr/bin/env python3
"""
Minimal Python Preprocessing Service for Image-to-SVG API
Only used for photo enhancement when quality is critical.
For logos, all processing stays in TypeScript/Sharp for speed.
"""

import sys
import cv2
import numpy as np
from typing import Optional
import argparse


class PhotoEnhancer:
    """
    Advanced photo preprocessing using OpenCV.
    NOT used for logos - only photographs and complex illustrations.
    """

    @staticmethod
    def denoise(
        input_path: str,
        output_path: str,
        strength: int = 10,
        preserve_details: bool = True
    ) -> None:
        """
        Non-local means denoising for JPEG artifact removal.
        Strength: 0-20 (higher = more smoothing)
        """
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")

        # Auto-adjust strength based on image size
        h, w = img.shape[:2]
        pixels = h * w
        
        # Reduce strength for very large images to prevent over-smoothing
        if pixels > 10_000_000:  # 10MP
            strength = max(3, strength // 2)

        if preserve_details and strength > 0:
            # Use faster method for large images
            if pixels > 5_000_000:
                img = cv2.fastNlMeansDenoisingColored(
                    img,
                    None,
                    h=strength,
                    hColor=strength,
                    templateWindowSize=7,
                    searchWindowSize=15  # Smaller = faster
                )
            else:
                img = cv2.fastNlMeansDenoisingColored(
                    img,
                    None,
                    h=strength,
                    hColor=strength,
                    templateWindowSize=7,
                    searchWindowSize=21
                )

        cv2.imwrite(output_path, img)

    @staticmethod
    def enhance_contrast(
        input_path: str,
        output_path: str,
        clip_limit: float = 3.0
    ) -> None:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Improves local contrast without noise amplification.
        """
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")

        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        tile_size = min(8, max(4, min(img.shape[:2]) // 256))
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
        l = clahe.apply(l)

        # Merge back
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        cv2.imwrite(output_path, img)

    @staticmethod
    def full_preprocess(
        input_path: str,
        output_path: str,
        noise_strength: int = 10,
        enhance: bool = True
    ) -> None:
        """
        Complete preprocessing pipeline for photos.
        Order matters: denoise first, then enhance.
        """
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")

        h, w = img.shape[:2]

        # Step 1: Resize if extremely large (memory protection)
        max_dim = 8192
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(
                img,
                (new_w, new_h),
                interpolation=cv2.INTER_LANCZOS4
            )
            h, w = new_h, new_w

        # Step 2: Denoise
        if noise_strength > 0:
            img = cv2.fastNlMeansDenoisingColored(
                img,
                None,
                h=noise_strength,
                hColor=noise_strength,
                templateWindowSize=7,
                searchWindowSize=15 if (h * w) > 5_000_000 else 21
            )

        # Step 3: Enhance contrast
        if enhance:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Step 4: Subtle sharpening (unsharp mask)
        gaussian = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

        cv2.imwrite(output_path, img)


def main():
    parser = argparse.ArgumentParser(
        description='Photo preprocessing for SVG conversion'
    )
    parser.add_argument('command', choices=[
        'denoise',
        'enhance',
        'full'
    ])
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--noise', type=int, default=10)
    parser.add_argument('--no-enhance', action='store_true')

    args = parser.parse_args()

    try:
        enhancer = PhotoEnhancer()

        if args.command == 'denoise':
            enhancer.denoise(args.input, args.output, args.noise)
        elif args.command == 'enhance':
            enhancer.enhance_contrast(args.input, args.output)
        elif args.command == 'full':
            enhancer.full_preprocess(
                args.input,
                args.output,
                args.noise,
                not args.no_enhance
            )

        print(f"SUCCESS: {args.output}", file=sys.stderr)
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()