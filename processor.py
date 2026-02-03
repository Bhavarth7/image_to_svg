#!/usr/bin/env python3
"""
High-Performance Image Preprocessing & AI Enhancement Service
Optional: Add StarVector for maximum quality on technical graphics

Install: pip install torch transformers accelerate opencv-python numpy Pillow
"""

import sys
import json
import base64
import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, Any
import argparse

class ImageProcessor:
    """Production-grade image preprocessing for vectorization"""
    
    def __init__(self, use_ai: bool = False):
        self.use_ai = use_ai
        self.model = None
        
        if use_ai:
            self._load_starvector()
    
    def _load_starvector(self):
        """Lazy load StarVector model for AI-enhanced vectorization"""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            self.processor = AutoProcessor.from_pretrained("starvector/starvector-8b-im2svg")
            self.model = AutoModelForVision2Seq.from_pretrained(
                "starvector/starvector-8b-im2svg",
                torch_dtype="auto",
                device_map="auto"
            )
            print("StarVector model loaded", file=sys.stderr)
        except Exception as e:
            print(f"AI model load failed: {e}", file=sys.stderr)
            self.use_ai = False
    
    def preprocess(self, image_bytes: bytes, params: Dict[str, Any]) -> bytes:
        """
        Preprocess pipeline:
        1. Decode and normalize
        2. Resize if needed
        3. Denoise (OpenCV)
        4. Enhance contrast/sharpness
        5. Return optimized PNG bytes
        """
        # Decode
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Resize if too large (keep under 4K for speed)
        max_dim = params.get('max_dimension', 4096)
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Denoise (JPEG artifact removal)
        noise_level = params.get('noise_reduction', 8)
        if noise_level > 0:
            # Adaptive denoising based on image size
            template_win = 7 if max(h, w) < 2000 else 5
            search_win = 21 if max(h, w) < 2000 else 15
            img = cv2.fastNlMeansDenoisingColored(
                img, None, 
                h=noise_level,
                hColor=noise_level,
                templateWindowSize=template_win,
                searchWindowSize=search_win
            )
        
        # Enhancement based on detail level
        detail = params.get('detail_level', 'high')
        if detail in ['high', 'maximum']:
            # CLAHE for local contrast
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            if detail == 'maximum':
                # Unsharp mask for edge enhancement
                gaussian = cv2.GaussianBlur(img, (0, 0), 3)
                img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        
        # Encode back to PNG
        success, encoded = cv2.imencode('.png', img)
        if not success:
            raise RuntimeError("Failed to encode image")
            
        return encoded.tobytes()
    
    def vectorize_ai(self, image_bytes: bytes) -> str:
        """Use StarVector for best quality vectorization (GPU required)"""
        if not self.model:
            raise RuntimeError("AI model not loaded")
        
        from transformers import AutoProcessor
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        
        # Generate SVG
        generated_ids = self.model.generate(**inputs, max_length=3000)
        svg_code = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return svg_code
    
    def optimize_svg(self, svg_content: str, params: Dict[str, Any]) -> str:
        """Post-process SVG for smaller file size"""
        # Remove unnecessary decimals
        import re
        
        if params.get('precision', 'standard') != 'premium':
            # Reduce coordinate precision to 2 decimal places
            svg_content = re.sub(r'(\d)\.(\d{3,})', r'\1.\2', svg_content)
            svg_content = re.sub(r'\.\d{3,}', lambda m: m.group(0)[:3], svg_content)
        
        # Remove empty groups
        svg_content = re.sub(r'<g[^>]*>\s*</g>', '', svg_content)
        
        return svg_content

# CLI Interface for TypeScript integration
def main():
    parser = argparse.ArgumentParser(description='Image preprocessing service')
    parser.add_argument('command', choices=['preprocess', 'vectorize', 'optimize'])
    parser.add_argument('--input', '-i', required=True, help='Input file path or base64')
    parser.add_argument('--output', '-o', required=True, help='Output file path')
    parser.add_argument('--params', '-p', default='{}', help='JSON params')
    parser.add_argument('--base64', '-b', action='store_true', help='Input is base64')
    parser.add_argument('--ai', '-a', action='store_true', help='Use AI vectorization')
    
    args = parser.parse_args()
    
    processor = ImageProcessor(use_ai=args.ai)
    params = json.loads(args.params)
    
    # Read input
    if args.base64:
        image_bytes = base64.b64decode(args.input)
    else:
        with open(args.input, 'rb') as f:
            image_bytes = f.read()
    
    # Process
    if args.command == 'preprocess':
        result = processor.preprocess(image_bytes, params)
        with open(args.output, 'wb') as f:
            f.write(result)
        print(json.dumps({"success": True, "output": args.output}))
        
    elif args.command == 'vectorize' and args.ai:
        svg = processor.vectorize_ai(image_bytes)
        optimized = processor.optimize_svg(svg, params)
        with open(args.output, 'w') as f:
            f.write(optimized)
        print(json.dumps({"success": True, "output": args.output}))
        
    elif args.command == 'optimize':
        with open(args.input, 'r') as f:
            svg = f.read()
        optimized = processor.optimize_svg(svg, params)
        with open(args.output, 'w') as f:
            f.write(optimized)
        print(json.dumps({"success": True, "output": args.output}))

if __name__ == '__main__':
    main()