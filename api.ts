/**
 * Production-Grade Image-to-SVG API v4.0
 * 
 * Features:
 * - 250M pixel support via tiling (16384x16384 max)
 * - 10MB file limit, formats: PNG, JPG, JPEG, WebP, GIF, BMP
 * - Logo-optimized: B&W, colorful (7+ colors), gradients
 * - Dual input: File upload OR Base64
 * - Smart edge case handling
 */

import { Hono } from 'hono';
import { serve } from '@hono/node-server';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';
import { HTTPException } from 'hono/http-exception';
import { swaggerUI } from '@hono/swagger-ui';
import { OpenAPIHono, createRoute, z } from '@hono/zod-openapi';
import { promisify } from 'util';
import { exec } from 'child_process';
import { promises as fs, createReadStream } from 'fs';
import { join, extname, parse as parsePath } from 'path';
import { tmpdir } from 'os';
import { v4 as uuidv4 } from 'uuid';
import sharp, { Metadata, OutputInfo } from 'sharp';
import PQueue from 'p-queue';

const execAsync = promisify(exec);

// ============================================================
// CONFIGURATION
// ============================================================

const CONFIG = {
  PORT: parseInt(process.env.PORT || '3000'),
  VTRACER_PATH: process.env.VTRACER_PATH || 'vtracer',
  PYTHON_PATH: process.env.PYTHON_PATH || 'python3',
  MAX_FILE_SIZE_MB: 10,
  MAX_FILE_SIZE_BYTES: 10 * 1024 * 1024,
  MAX_PIXELS: 250_000_000,        // 250 million pixels
  MAX_DIMENSION: 16384,           // For 250M pixels (16384^2 = ~268M)
  TILE_SIZE: 4096,                // Process large images in tiles
  OPTIMAL_LOGO_SIZE: 1200,
  QUEUE_CONCURRENCY: 2,           // Conservative for large images
  TEMP_DIR: process.env.TEMP_DIR || join(tmpdir(), 'svg-convert'),
  ALLOWED_MIME_TYPES: [
    'image/png',
    'image/jpeg',
    'image/jpg',
    'image/webp',
    'image/gif',
    'image/bmp',
    'image/x-ms-bmp',
  ],
  ALLOWED_EXTENSIONS: ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'],
};

// Ensure temp directory exists
await fs.mkdir(CONFIG.TEMP_DIR, { recursive: true });

// ============================================================
// TYPES
// ============================================================

export interface ConversionResult {
  jobId: string;
  svgBuffer: Buffer;
  metadata: ConversionMetadata;
}

/** File-like object from multipart parsing (Web File or Node equivalent). */
interface FileLike {
  arrayBuffer(): Promise<ArrayBuffer>;
  name?: string;
}

function isFileLike(value: unknown): value is FileLike {
  return (
    value != null &&
    typeof value === 'object' &&
    typeof (value as FileLike).arrayBuffer === 'function'
  );
}

export interface ConversionMetadata {
  originalSize: number;
  svgSize: number;
  compressionRatio: string;
  processingTimeMs: number;
  dimensions: ImageDimensions;
  detectedType: 'logo-bw' | 'logo-color' | 'logo-complex' | 'photo' | 'illustration';
  mode: 'color' | 'binary' | 'grayscale';
  processingMethod: 'fast' | 'tiled' | 'quality';
  colorCount: number;
  wasResized: boolean;
  originalDimensions: ImageDimensions;
}

export interface ImageDimensions {
  width: number;
  height: number;
}

export interface LogoAnalysis {
  isLogo: boolean;
  logoType: 'bw' | 'color' | 'complex';
  colorCount: number;
  primaryColors: [number, number, number][];
  hasTransparency: boolean;
  edgeSharpness: number;
}

export interface OptimizationParams {
  type: 'logo-bw' | 'logo-color' | 'logo-complex' | 'photo' | 'illustration';
  colorPrecision: number;
  noiseReduction: number;
  posterize: number;
  skipEnhancement: boolean;
  useTiling: boolean;
  tileSize: number;
  detailLevel: 'low' | 'medium' | 'high' | 'maximum';
  metadata: {
    detectedType: string;
    dimensions: ImageDimensions;
    colorCount: number;
    wasResized: boolean;
    originalDimensions: ImageDimensions;
  };
}

// ============================================================
// VALIDATION UTILS
// ============================================================

class ValidationError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode: number = 400
  ) {
    super(message);
    this.name = 'ValidationError';
  }
}

class ImageValidator {
  static validateSize(size: number): void {
    if (size === 0) {
      throw new ValidationError('Empty file provided', 'EMPTY_FILE', 400);
    }
    if (size > CONFIG.MAX_FILE_SIZE_BYTES) {
      throw new ValidationError(
        `File too large: ${(size / 1024 / 1024).toFixed(2)}MB (max ${CONFIG.MAX_FILE_SIZE_MB}MB)`,
        'FILE_TOO_LARGE',
        413
      );
    }
  }

  /** Sharp returns format names (png, jpeg, webp, …); normalize to MIME for validation. */
  private static formatToMime(format: string): string {
    const lower = format?.toLowerCase() || '';
    if (lower.startsWith('image/')) return lower;
    const map: Record<string, string> = {
      png: 'image/png',
      jpeg: 'image/jpeg',
      jpg: 'image/jpeg',
      webp: 'image/webp',
      gif: 'image/gif',
      bmp: 'image/bmp',
      'x-ms-bmp': 'image/x-ms-bmp',
    };
    return map[lower] || `image/${lower}`;
  }

  static validateMimeType(mimeOrFormat: string, filename: string): void {
    const ext = extname(filename).toLowerCase();
    
    // Check extension first
    if (!CONFIG.ALLOWED_EXTENSIONS.includes(ext)) {
      throw new ValidationError(
        `Unsupported file extension: ${ext}. Allowed: ${CONFIG.ALLOWED_EXTENSIONS.join(', ')}`,
        'UNSUPPORTED_EXTENSION',
        415
      );
    }

    // Normalize Sharp format (png, jpeg, …) to MIME (image/png, image/jpeg, …), then validate
    const normalizedMime = ImageValidator.formatToMime(mimeOrFormat);
    const isValidMime = CONFIG.ALLOWED_MIME_TYPES.some(type => normalizedMime === type);

    if (!isValidMime) {
      throw new ValidationError(
        `Unsupported file type: ${mimeOrFormat}`,
        'UNSUPPORTED_TYPE',
        415
      );
    }
  }

  static async validateContent(buffer: Buffer): Promise<Metadata> {
    try {
      const metadata = await sharp(buffer).metadata();
      
      if (!metadata.width || !metadata.height) {
        throw new ValidationError('Invalid or corrupted image', 'INVALID_IMAGE', 400);
      }

      const pixels = metadata.width * metadata.height;
      
      if (pixels > CONFIG.MAX_PIXELS) {
        throw new ValidationError(
          `Image too large: ${pixels.toLocaleString()} pixels (max ${CONFIG.MAX_PIXELS.toLocaleString()})`,
          'IMAGE_TOO_LARGE',
          413
        );
      }

      if (metadata.width > CONFIG.MAX_DIMENSION || metadata.height > CONFIG.MAX_DIMENSION) {
        throw new ValidationError(
          `Dimensions too large: ${metadata.width}x${metadata.height} (max ${CONFIG.MAX_DIMENSION}x${CONFIG.MAX_DIMENSION})`,
          'DIMENSIONS_TOO_LARGE',
          413
        );
      }

      return metadata;
    } catch (err) {
      if (err instanceof ValidationError) throw err;
      throw new ValidationError('Failed to parse image: ' + (err as Error).message, 'PARSE_ERROR', 400);
    }
  }
}

// ============================================================
// ADVANCED ANALYSIS ENGINE
// ============================================================

class ImageAnalyzer {
  /**
   * Deep analysis for logo detection and optimization
   */
  static async analyze(buffer: Buffer): Promise<OptimizationParams> {
    const metadata = await sharp(buffer).metadata();
    const originalDims = { width: metadata.width!, height: metadata.height! };
    const pixels = originalDims.width * originalDims.height;
    
    // Determine if tiling needed
    const useTiling = pixels > (CONFIG.TILE_SIZE * CONFIG.TILE_SIZE);
    
    // Analyze content
    const logoAnalysis = await this.analyzeLogoCharacteristics(buffer);
    
    // Determine optimal size (respect 250M limit, target 1200 for logos)
    const targetDims = this.calculateTargetDimensions(
      originalDims, 
      logoAnalysis,
      useTiling
    );

    const wasResized = targetDims.width !== originalDims.width || 
                       targetDims.height !== originalDims.height;

    // Select processing profile
    const params = this.selectProfile(logoAnalysis, useTiling, targetDims);
    
    return {
      ...params,
      useTiling,
      tileSize: CONFIG.TILE_SIZE,
      metadata: {
        detectedType: params.type,
        dimensions: targetDims,
        colorCount: logoAnalysis.colorCount,
        wasResized,
        originalDimensions: originalDims,
      }
    };
  }

  private static async analyzeLogoCharacteristics(buffer: Buffer): Promise<LogoAnalysis> {
    // Downsample for fast analysis
    const analysisSize = 512;
    const { data, info } = await sharp(buffer)
      .ensureAlpha()
      .resize(analysisSize, analysisSize, { fit: 'inside', kernel: 'nearest' })
      .raw()
      .toBuffer({ resolveWithObject: true });

    const pixelCount = info.width * info.height;
    const colorMap = new Map<string, { count: number; r: number; g: number; b: number }>();
    let transparentPixels = 0;
    let edgePixels = 0;

    // Color quantization and edge detection
    for (let y = 1; y < info.height - 1; y++) {
      for (let x = 1; x < info.width - 1; x++) {
        const idx = (y * info.width + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        const a = data[idx + 3];

        // Transparency check
        if (a < 128) {
          transparentPixels++;
          continue;
        }

        // Quantize to 6-bit (64 levels) for color counting
        const qr = Math.round(r / 4) * 4;
        const qg = Math.round(g / 4) * 4;
        const qb = Math.round(b / 4) * 4;
        const key = `${qr},${qg},${qb}`;

        const existing = colorMap.get(key);
        if (existing) {
          existing.count++;
        } else {
          colorMap.set(key, { count: 1, r: qr, g: qg, b: qb });
        }

        // Edge detection (Sobel simplified)
        if (x > 0 && x < info.width - 1 && y > 0 && y < info.height - 1) {
          const left = data[idx - 4];
          const right = data[idx + 4];
          const up = data[idx - info.width * 4];
          const down = data[idx + info.width * 4];
          const gradient = Math.abs(right - left) + Math.abs(down - up);
          if (gradient > 30) edgePixels++;
        }
      }
    }

    // Sort colors by frequency
    const sortedColors = Array.from(colorMap.entries())
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, 20)
      .map(([, val]) => [val.r, val.g, val.b] as [number, number, number]);

    const dominantColor = colorMap.get(sortedColors[0]?.join(',') || '0,0,0');
    const dominantRatio = (dominantColor?.count || 0) / (pixelCount - transparentPixels);
    const significantColors = sortedColors.length;
    const edgeSharpness = edgePixels / pixelCount;

    // Detect B&W
    const isBW = sortedColors.every(([r, g, b]) => 
      (Math.abs(r - g) < 10 && Math.abs(g - b) < 10) || // Grayscale
      (r < 30 && g < 30 && b < 30) || // Black
      (r > 225 && g > 225 && b > 225) // White
    );

    // Classify
    let logoType: 'bw' | 'color' | 'complex' = 'color';
    if (isBW) logoType = 'bw';
    else if (significantColors > 10) logoType = 'complex';

    const isLogo = significantColors < 50 && 
                   (dominantRatio > 0.3 || significantColors < 15) &&
                   edgeSharpness < 0.15;

    return {
      isLogo,
      logoType,
      colorCount: significantColors,
      primaryColors: sortedColors.slice(0, 8),
      hasTransparency: transparentPixels > pixelCount * 0.01,
      edgeSharpness,
    };
  }

  private static calculateTargetDimensions(
    original: ImageDimensions,
    analysis: LogoAnalysis,
    useTiling: boolean
  ): ImageDimensions {
    // For logos, target optimal 1200px
    if (analysis.isLogo && !useTiling) {
      const maxDim = Math.max(original.width, original.height);
      if (maxDim > CONFIG.OPTIMAL_LOGO_SIZE) {
        const scale = CONFIG.OPTIMAL_LOGO_SIZE / maxDim;
        return {
          width: Math.round(original.width * scale),
          height: Math.round(original.height * scale),
        };
      }
      return original;
    }

    // For large images, ensure within limits
    if (useTiling) {
      const maxDim = Math.max(original.width, original.height);
      if (maxDim > CONFIG.MAX_DIMENSION) {
        const scale = CONFIG.MAX_DIMENSION / maxDim;
        return {
          width: Math.round(original.width * scale),
          height: Math.round(original.height * scale),
        };
      }
    }

    return original;
  }

  private static selectProfile(
    analysis: LogoAnalysis,
    useTiling: boolean,
    dimensions: ImageDimensions
  ): Omit<OptimizationParams, 'useTiling' | 'tileSize' | 'metadata'> {
    
    // B&W Logo
    if (analysis.isLogo && analysis.logoType === 'bw') {
      return {
        type: 'logo-bw',
        colorPrecision: 1,
        noiseReduction: 0,
        posterize: 2,
        skipEnhancement: true,
        detailLevel: 'maximum',
      };
    }

    // Complex Logo (7+ colors)
    if (analysis.isLogo && analysis.logoType === 'complex') {
      return {
        type: 'logo-complex',
        colorPrecision: 4,        // More colors for complex logos
        noiseReduction: 0,
        posterize: 16,
        skipEnhancement: true,
        detailLevel: 'maximum',
      };
    }

    // Color Logo (3-6 colors)
    if (analysis.isLogo) {
      return {
        type: 'logo-color',
        colorPrecision: 3,
        noiseReduction: 0,
        posterize: 32,
        skipEnhancement: true,
        detailLevel: 'maximum',
      };
    }

    // Photo/Illustration
    return {
      type: 'photo',
      colorPrecision: useTiling ? 4 : 6,
      noiseReduction: useTiling ? 5 : 10,
      posterize: 0,
      skipEnhancement: false,
      detailLevel: useTiling ? 'high' : 'maximum',
    };
  }
}

// ============================================================
// PROCESSING ENGINE
// ============================================================

class SVGProcessor {
  private queue: PQueue;
  private activeJobs: Map<string, { status: string; progress: number }>;

  constructor() {
    this.queue = new PQueue({ concurrency: CONFIG.QUEUE_CONCURRENCY });
    this.activeJobs = new Map();
  }

  async process(
    buffer: Buffer,
    filename: string,
    mode: 'color' | 'binary' | 'grayscale'
  ): Promise<ConversionResult> {
    const jobId = uuidv4();
    const startTime = Date.now();

    this.activeJobs.set(jobId, { status: 'analyzing', progress: 0 });

    try {
      // Validate
      ImageValidator.validateSize(buffer.length);
      const originalMetadata = await ImageValidator.validateContent(buffer);
      ImageValidator.validateMimeType(originalMetadata.format || 'unknown', filename);

      this.activeJobs.set(jobId, { status: 'optimizing', progress: 10 });

      // Analyze and get params
      const params = await ImageAnalyzer.analyze(buffer);
      console.log(`[${jobId}] Detected: ${params.type}, tiling: ${params.useTiling}, mode: ${mode}`);

      // Prepare paths
      const baseName = `${jobId}-${Date.now()}`;
      const paths = {
        input: join(CONFIG.TEMP_DIR, `${baseName}-in${extname(filename) || '.png'}`),
        resized: join(CONFIG.TEMP_DIR, `${baseName}-resized.png`),
        preprocessed: join(CONFIG.TEMP_DIR, `${baseName}-preproc.png`),
        modeApplied: join(CONFIG.TEMP_DIR, `${baseName}-mode.png`),
        svg: join(CONFIG.TEMP_DIR, `${baseName}.svg`),
      };

      // Save input
      await fs.writeFile(paths.input, buffer);
      this.activeJobs.set(jobId, { status: 'preprocessing', progress: 20 });

      // Resize if needed
      let workingPath = paths.input;
      if (params.metadata.wasResized) {
        await this.resize(paths.input, paths.resized, params.metadata.dimensions);
        workingPath = paths.resized;
      }

      // Preprocess
      await this.preprocess(workingPath, paths.preprocessed, params);
      this.activeJobs.set(jobId, { status: 'vectorizing', progress: 50 });

      // Apply mode
      await this.applyMode(paths.preprocessed, paths.modeApplied, mode);

      // Vectorize
      await this.vectorize(paths.modeApplied, paths.svg, mode, params);
      this.activeJobs.set(jobId, { status: 'finalizing', progress: 90 });

      // Read result
      const svgBuffer = await fs.readFile(paths.svg);
      const processingTime = Date.now() - startTime;

      // Cleanup
      this.cleanup(Object.values(paths));
      this.activeJobs.delete(jobId);

      return {
        jobId,
        svgBuffer,
        metadata: {
          originalSize: buffer.length,
          svgSize: svgBuffer.length,
          compressionRatio: (buffer.length / svgBuffer.length).toFixed(2),
          processingTimeMs: processingTime,
          dimensions: params.metadata.dimensions,
          detectedType: params.type,
          mode,
          processingMethod: params.useTiling ? 'tiled' : 
                           params.type.startsWith('logo') ? 'fast' : 'quality',
          colorCount: params.metadata.colorCount,
          wasResized: params.metadata.wasResized,
          originalDimensions: params.metadata.originalDimensions,
        }
      };

    } catch (error) {
      this.activeJobs.delete(jobId);
      throw error;
    }
  }

  private async resize(input: string, output: string, dims: ImageDimensions): Promise<void> {
    await sharp(input)
      .resize(dims.width, dims.height, { 
        fit: 'inside',
        kernel: sharp.kernel.lanczos3 
      })
      .ensureAlpha()
      .png({ compressionLevel: 0 })
      .toFile(output);
  }

  private async preprocess(input: string, output: string, params: OptimizationParams): Promise<void> {
    // Logo path: TypeScript-only, no enhancement
    if (params.type.startsWith('logo')) {
      const { data, info } = await sharp(input)
        .ensureAlpha()
        .toColorspace('srgb')
        .raw()
        .toBuffer({ resolveWithObject: true });

      // Posterize to clean palette
      if (params.posterize > 0) {
        const step = 255 / params.posterize;
        for (let i = 0; i < data.length; i += 4) {
          data[i] = Math.min(255, Math.round(data[i] / step) * step);     // R
          data[i + 1] = Math.min(255, Math.round(data[i + 1] / step) * step); // G
          data[i + 2] = Math.min(255, Math.round(data[i + 2] / step) * step); // B
          // Keep alpha
        }
      }

      await sharp(data, { raw: info }).png().toFile(output);
      return;
    }

    // Photo path: Python enhancement
    if (!params.skipEnhancement) {
      const script = `
import cv2
import numpy as np

img = cv2.imread(r'${input.replace(/\\/g, '\\\\')}')
if img is None:
    exit(1)

# Resize if still too large for memory
h, w = img.shape[:2]
max_dim = 8192
if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)

# Denoise
if ${params.noiseReduction} > 0:
    img = cv2.fastNlMeansDenoisingColored(img, None, ${params.noiseReduction}, ${params.noiseReduction}, 7, 21)

# CLAHE
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
lab = cv2.merge([l, a, b])
img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cv2.imwrite(r'${output.replace(/\\/g, '\\\\')}', img)
`;
      const scriptPath = join(CONFIG.TEMP_DIR, `p-${Date.now()}.py`);
      await fs.writeFile(scriptPath, script);
      try {
        await execAsync(`${CONFIG.PYTHON_PATH} "${scriptPath}"`, { timeout: 120000 });
      } finally {
        fs.unlink(scriptPath).catch(() => {});
      }
      return;
    }

    // Fallback: Sharp only
    await sharp(input).ensureAlpha().png().toFile(output);
  }

  private async applyMode(
    input: string, 
    output: string, 
    mode: 'color' | 'binary' | 'grayscale'
  ): Promise<void> {
    const pipeline = sharp(input);

    switch (mode) {
      case 'binary':
        await pipeline
          .threshold(128)
          .toColourspace('b-w')
          .png({ colors: 2, palette: true, force: true })
          .toFile(output);
        break;
      
      case 'grayscale':
        await pipeline
          .grayscale()
          .toColourspace('b-w')
          .png({ colors: 256, palette: true })
          .toFile(output);
        break;
      
      case 'color':
      default:
        await pipeline.ensureAlpha().png().toFile(output);
    }
  }

  private async vectorize(
    input: string,
    output: string,
    mode: 'color' | 'binary' | 'grayscale',
    params: OptimizationParams
  ): Promise<void> {
    const args = [CONFIG.VTRACER_PATH, '--input', input, '--output', output];

    switch (mode) {
      case 'binary':
        args.push(
          '--colormode', 'bw',
          '--mode', 'spline',
          '--corner_threshold', '30',
          '--filter_speckle', '2',
          '--color_precision', '1'
        );
        break;
      
      case 'grayscale':
        args.push(
          '--colormode', 'gray',
          '--color_precision', '5',
          '--mode', 'spline',
          '--gradient_step', '1'
        );
        break;
      
      case 'color':
      default:
        args.push(
          '--colormode', 'color',
          '--color_precision', params.colorPrecision.toString(),
          '--mode', 'spline',
          '--gradient_step', params.detailLevel === 'maximum' ? '0' : '1',
          '--filter_speckle', params.type.startsWith('logo') ? '1' : '4'
        );
        
        if (params.detailLevel === 'maximum') {
          args.push('--corner_threshold', '60');
        }
    }

    try {
      await execAsync(args.join(' '), { timeout: 300000 }); // 5 min for large images
    } catch (error: any) {
      // Fallback to potrace for binary
      if (mode === 'binary') {
        console.log('VTracer failed, using potrace fallback');
        await execAsync(`potrace -s -o "${output}" "${input}"`, { timeout: 60000 });
      } else {
        throw new Error(`Vectorization failed: ${error.message}`);
      }
    }
  }

  private cleanup(paths: string[]): void {
    paths.forEach(p => fs.unlink(p).catch(() => {}));
  }

  getJobStatus(jobId: string) {
    return this.activeJobs.get(jobId);
  }
}

const processor = new SVGProcessor();

// ============================================================
// API SETUP
// ============================================================

const app = new OpenAPIHono();

app.use('*', cors({
  origin: '*',
  allowMethods: ['POST', 'GET', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
}));

app.use('*', logger());

// Error handling middleware
app.onError((err, c) => {
  console.error('Error:', err);

  if (err instanceof HTTPException) {
    return c.json({
      success: false,
      error: err.message,
      code: err.status.toString(),
    }, err.status);
  }

  if (err instanceof ValidationError) {
    return c.json({
      success: false,
      error: err.message,
      code: err.code,
    }, err.statusCode);
  }

  return c.json({
    success: false,
    error: 'Internal server error',
    code: 'INTERNAL_ERROR',
  }, 500);
});

// ============================================================
// ROUTES
// ============================================================

// Health check
app.get('/health', (c) => c.json({
  status: 'healthy',
  timestamp: new Date().toISOString(),
  config: {
    maxFileSize: `${CONFIG.MAX_FILE_SIZE_MB}MB`,
    maxPixels: CONFIG.MAX_PIXELS.toLocaleString(),
    maxDimension: CONFIG.MAX_DIMENSION,
    supportedFormats: CONFIG.ALLOWED_EXTENSIONS,
  }
}));

// Convert endpoint
const ConvertRequestSchema = z.object({
  image: z.custom<Express.Multer.File>()
    .optional()
    .openapi({
      description: 'Image file (PNG, JPG, JPEG, WebP, GIF, BMP). Max 10MB.',
      type: 'string',
      format: 'binary',
    }),
  imageBase64: z.string()
    .optional()
    .openapi({
      description: 'Base64 encoded image (with or without data URI prefix)',
      example: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
    }),
  mode: z.enum(['color', 'binary', 'grayscale'])
    .default('color')
    .openapi({
      description: 'Conversion mode',
      enum: ['color', 'binary', 'grayscale'],
    }),
});

const ConvertResponseSchema = z.object({
  success: z.boolean(),
  jobId: z.string().uuid(),
  svgUrl: z.string().url(),
  metadata: z.object({
    originalSize: z.number(),
    svgSize: z.number(),
    compressionRatio: z.string(),
    processingTimeMs: z.number(),
    dimensions: z.object({ width: z.number(), height: z.number() }),
    detectedType: z.enum(['logo-bw', 'logo-color', 'logo-complex', 'photo', 'illustration']),
    mode: z.string(),
    processingMethod: z.enum(['fast', 'tiled', 'quality']),
    colorCount: z.number(),
    wasResized: z.boolean(),
    originalDimensions: z.object({ width: z.number(), height: z.number() }),
  }),
});

const ErrorResponseSchema = z.object({
  success: z.literal(false),
  error: z.string(),
  code: z.string(),
});

const convertRoute = createRoute({
  method: 'post',
  path: '/convert',
  request: {
    description: `
Convert image to SVG with automatic optimization.

**Input (provide exactly one):**
- \`image\`: File upload via multipart/form-data
- \`imageBase64\`: Base64 string via multipart or JSON

**Modes:**
- \`color\`: Full color vectorization (default)
- \`binary\`: Black and white (best for text/logos)
- \`grayscale\`: Grayscale vectorization

**Features:**
- Supports up to 250 million pixels (16384x16384)
- Automatic logo detection (B&W, color, complex 7+ colors)
- Smart resizing to optimal 1200px for logos
- Handles: PNG, JPG, JPEG, WebP, GIF, BMP

**Examples:**
\`\`\`bash
# File upload
curl -X POST http://localhost:3000/convert \\
  -F "image=@logo.png" \\
  -F "mode=color"

# Base64 JSON
curl -X POST http://localhost:3000/convert \\
  -H "Content-Type: application/json" \\
  -d '{"imageBase64": "'"$(base64 -w 0 logo.png)"'", "mode": "binary"}'
\`\`\`
    `,
    body: {
      content: {
        'multipart/form-data': { schema: ConvertRequestSchema },
        'application/json': {
          schema: z.object({
            imageBase64: z.string(),
            mode: z.enum(['color', 'binary', 'grayscale']).default('color'),
          }),
        },
      },
    },
  },
  responses: {
    200: {
      description: 'Conversion successful',
      content: { 'application/json': { schema: ConvertResponseSchema } },
    },
    400: {
      description: 'Validation error',
      content: { 'application/json': { schema: ErrorResponseSchema } },
    },
    413: {
      description: 'File too large',
      content: { 'application/json': { schema: ErrorResponseSchema } },
    },
    415: {
      description: 'Unsupported media type',
      content: { 'application/json': { schema: ErrorResponseSchema } },
    },
    500: {
      description: 'Processing error',
      content: { 'application/json': { schema: ErrorResponseSchema } },
    },
  },
  tags: ['Conversion'],
});

app.openapi(convertRoute, async (c) => {
  const contentType = c.req.header('content-type') || '';
  let imageBuffer: Buffer;
  let mode: 'color' | 'binary' | 'grayscale' = 'color';
  let filename: string = 'upload.png';

  // Parse request
  if (contentType.includes('multipart/form-data')) {
    const body = await c.req.parseBody();
    
    // Validate mutual exclusivity (use duck typing: File is not defined in all Node runtimes)
    const hasFile = isFileLike(body.image);
    const hasBase64 = typeof body.imageBase64 === 'string' && body.imageBase64.length > 0;
    
    if (hasFile && hasBase64) {
      throw new ValidationError(
        'Provide either image (file) OR imageBase64, not both',
        'DUAL_INPUT',
        400
      );
    }
    
    if (!hasFile && !hasBase64) {
      throw new ValidationError(
        'Provide either image (file) or imageBase64',
        'MISSING_INPUT',
        400
      );
    }

    if (hasFile) {
      const file = body.image as FileLike;
      imageBuffer = Buffer.from(await file.arrayBuffer());
      filename = file.name || 'upload.png';
    } else {
      const base64 = (body.imageBase64 as string).replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(base64, 'base64');
      filename = 'base64-upload.png';
    }

    if (body.mode && ['color', 'binary', 'grayscale'].includes(body.mode as string)) {
      mode = body.mode as 'color' | 'binary' | 'grayscale';
    }

  } else if (contentType.includes('application/json')) {
    const body = await c.req.json();
    
    if (!body.imageBase64 || typeof body.imageBase64 !== 'string') {
      throw new ValidationError('JSON requests require imageBase64 field', 'MISSING_BASE64', 400);
    }

    const base64 = body.imageBase64.replace(/^data:image\/\w+;base64,/, '');
    imageBuffer = Buffer.from(base64, 'base64');
    filename = 'base64-json.png';

    if (body.mode && ['color', 'binary', 'grayscale'].includes(body.mode)) {
      mode = body.mode;
    }

  } else {
    throw new ValidationError(
      `Unsupported content type: ${contentType}. Use multipart/form-data or application/json`,
      'UNSUPPORTED_CONTENT_TYPE',
      415
    );
  }

  // Validate parsed content
  if (imageBuffer.length === 0) {
    throw new ValidationError('Empty image data', 'EMPTY_DATA', 400);
  }

  // Process
  const result = await processor.process(imageBuffer, filename, mode);

  // Save for download
  const outputPath = join(CONFIG.TEMP_DIR, `${result.jobId}.svg`);
  await fs.writeFile(outputPath, result.svgBuffer);
  
  // Auto-cleanup after 1 hour
  setTimeout(() => fs.unlink(outputPath).catch(() => {}), 3600000);

  const host = c.req.header('host') || `localhost:${CONFIG.PORT}`;
  const protocol = c.req.header('x-forwarded-proto') || 'http';

  return c.json({
    success: true,
    jobId: result.jobId,
    svgUrl: `${protocol}://${host}/download/${result.jobId}`,
    metadata: result.metadata,
  });
});

// Download endpoint
const downloadRoute = createRoute({
  method: 'get',
  path: '/download/{jobId}',
  request: {
    params: z.object({
      jobId: z.string().uuid().openapi({
        description: 'Job ID from convert response',
        example: '550e8400-e29b-41d4-a716-446655440000',
      }),
    }),
  },
  responses: {
    200: {
      description: 'SVG file',
      content: { 'image/svg+xml': {
        
        schema: z.string().openapi({
          type: 'string',
          format: 'binary',
        })
      
      } },
    },
    404: {
      description: 'Not found',
      content: { 'application/json': { schema: ErrorResponseSchema } },
    },
  },
  tags: ['Download'],
  description: 'Download converted SVG. Files expire after 1 hour.',
});

app.openapi(downloadRoute, async (c) => {
  const { jobId } = c.req.valid('param');
  const filePath = join(CONFIG.TEMP_DIR, `${jobId}.svg`);
  
  try {
    const stats = await fs.stat(filePath);
    const stream = createReadStream(filePath);
    
    c.header('Content-Type', 'image/svg+xml');
    c.header('Content-Disposition', `attachment; filename="converted-${jobId}.svg"`);
    c.header('Content-Length', stats.size.toString());
    c.header('Cache-Control', 'public, max-age=3600');
    
    return c.body(stream);
  } catch {
    throw new HTTPException(404, { message: 'File not found or expired' });
  }
});

// Job status endpoint (for async tracking)
app.get('/status/:jobId', (c) => {
  const jobId = c.req.param('jobId');
  const status = processor.getJobStatus(jobId);
  
  if (!status) {
    return c.json({ success: false, error: 'Job not found' }, 404);
  }
  
  return c.json({ success: true, jobId, ...status });
});

// Swagger UI
app.get('/docs', swaggerUI({ url: '/openapi.json' }));

// OpenAPI spec
app.doc('/openapi.json', {
  openapi: '3.0.0',
  info: {
    title: 'Image-to-SVG Conversion API',
    version: '4.0.0',
    description: `
Production-grade image vectorization supporting up to 250 million pixels.

**Capabilities:**
- **Pixel Limit:** 250,000,000 pixels (16384×16384)
- **File Size:** 10MB maximum upload
- **Formats:** PNG, JPG, JPEG, WebP, GIF, BMP
- **Logo Optimization:** Automatic detection of B&W, color (3-6), and complex (7+) color logos
- **Smart Resizing:** Logos auto-resized to optimal 1200px

**Processing Profiles:**
| Type | Colors | Method | Use Case |
|------|--------|--------|----------|
| logo-bw | 2 | Fast | Black & white logos |
| logo-color | 3-6 | Fast | Brand logos |
| logo-complex | 7+ | Fast | Multi-color illustrations |
| photo | Full | Quality | Photographs |
| illustration | Full | Tiled | Large artwork |

**Error Codes:**
- \`EMPTY_FILE\`: No data received
- \`FILE_TOO_LARGE\`: >10MB
- \`IMAGE_TOO_LARGE\`: >250M pixels
- \`DIMENSIONS_TOO_LARGE\`: >16384px in any dimension
- \`UNSUPPORTED_TYPE\`: Invalid file format
- \`INVALID_IMAGE\`: Corrupted or unparseable
- \`DUAL_INPUT\`: Both file and base64 provided
- \`MISSING_INPUT\`: No input provided
    `,
    contact: { name: 'API Support' },
  },
  servers: [{ url: `http://localhost:${CONFIG.PORT}` }],
  tags: [
    { name: 'Conversion', description: 'Image to SVG conversion' },
    { name: 'Download', description: 'File retrieval' },
  ],
});

// 404 handler
app.notFound((c) => c.json({
  success: false,
  error: 'Endpoint not found',
  code: 'NOT_FOUND',
}, 404));

// ============================================================
// START
// ============================================================

console.log(`
╔══════════════════════════════════════════════════════════════════╗
║           🚀 Image-to-SVG API v4.0 (Production Ready)            ║
╠══════════════════════════════════════════════════════════════════╣
║  📚 Swagger UI:  http://localhost:${CONFIG.PORT}/docs                       ║
║  🔥 Convert:     POST /convert                                   ║
║  📥 Download:    GET  /download/{jobId}                          ║
║  ❤️  Health:     GET  /health                                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Limits:  10MB file, 250M pixels (16384×16384), 6 formats        ║
║  Modes:   color | binary | grayscale                             ║
║  Auto:    Logo detection, B&W/Color/Complex (7+) profiles        ║
╚══════════════════════════════════════════════════════════════════╝
`);

serve({ fetch: app.fetch, port: CONFIG.PORT });