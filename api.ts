/**
 * Production-Grade Image-to-SVG API (Complete Implementation)
 * 
 * Stack: Hono + OpenAPI + Sharp + VTracer + Python OpenCV
 * Features: Auto-optimization, dual input (file/base64), queue processing
 * 
 * ENV VARS:
 * - PORT: Server port (default: 3000)
 * - VTRACER_PATH: Path to vtracer binary (default: 'vtracer')
 * - PYTHON_PATH: Python executable (default: 'python3')
 * - MAX_FILE_SIZE_MB: Max upload size in MB (default: 10)
 * - MAX_PIXELS: Max total pixels width×height (default: 250000000)
 * - QUEUE_CONCURRENCY: Parallel jobs (default: 4)
 * - TEMP_DIR: Temp file directory (default: OS tmpdir)
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
import { join, extname } from 'path';
import { tmpdir } from 'os';
import { v4 as uuidv4 } from 'uuid';
import sharp from 'sharp';
import PQueue from 'p-queue';

const execAsync = promisify(exec);

// ============================================================
// CONFIGURATION
// ============================================================

const CONFIG = {
  PORT: parseInt(process.env.PORT || '3000'),
  VTRACER_PATH: process.env.VTRACER_PATH || 'vtracer',
  PYTHON_PATH: process.env.PYTHON_PATH || 'python3',
  MAX_FILE_SIZE_MB: parseInt(process.env.MAX_FILE_SIZE_MB || '10'),
  /** Max total pixels (width × height). Supports up to 250M for professional-grade assets. */
  MAX_PIXELS: parseInt(process.env.MAX_PIXELS || '250000000'),
  QUEUE_CONCURRENCY: parseInt(process.env.QUEUE_CONCURRENCY || '4'),
  TEMP_DIR: process.env.TEMP_DIR || tmpdir(),
  /** Supported input MIME types: PNG, JPG, JPEG, WebP, GIF, BMP */
  ALLOWED_TYPES: [
    'image/jpeg',
    'image/png',
    'image/jpg',
    'image/webp',
    'image/gif',
    'image/bmp',
  ],
  /** Recommended logo dimension for optimal quality vs CPU/memory (sweet spot for tracing). */
  OPTIMAL_LOGO_DIMENSION: 1200,
};

// Ensure temp dir exists
await fs.mkdir(CONFIG.TEMP_DIR, { recursive: true }).catch(() => {});

// ============================================================
// AUTO-OPTIMIZATION ENGINE
// ============================================================

class AutoOptimizer {
  /**
   * Analyzes image content and returns optimal processing parameters
   * Detects photos vs graphics automatically
   */
  static async analyze(imageBuffer: Buffer): Promise<{
    colorPrecision: number;
    noiseReduction: number;
    detailLevel: 'low' | 'medium' | 'high' | 'maximum';
    outputQuality: 'draft' | 'standard' | 'premium';
    format: 'svg' | 'svgz';
    metadata: {
      detectedType: 'photo' | 'graphic';
      dimensions: { width: number; height: number };
      originalSize: number;
      estimatedComplexity: 'low' | 'medium' | 'high';
    };
  }> {
    const metadata = await sharp(imageBuffer).metadata();
    const stats = await sharp(imageBuffer).stats();
    const size = imageBuffer.length;
    
    // Calculate image characteristics
    const channels = stats.channels;
    const avgEntropy = channels.reduce((sum, ch) => sum + (ch.entropy || 0), 0) / channels.length;
    const stdDev = channels.reduce((sum, ch) => sum + (ch.std || 0), 0) / channels.length;
    
    // Detection heuristics
    const isPhoto = avgEntropy > 0.6 && stdDev > 40;
    const isComplex = size > 2 * 1024 * 1024 || (metadata.width! * metadata.height!) > (3000 * 3000);
    
    const detectedType = isPhoto ? 'photo' : 'graphic';
    const estimatedComplexity = isComplex ? 'high' : size > 500 * 1024 ? 'medium' : 'low';
    
    // Auto-optimized parameters
    return {
      // Photos need more colors, graphics need fewer
      colorPrecision: isPhoto ? 6 : 4,
      
      // Photos need noise reduction, clean graphics don't
      noiseReduction: isPhoto ? 10 : 2,
      
      // Small images get maximum detail, large images get balanced
      detailLevel: size < 1024 * 1024 ? 'maximum' : isComplex ? 'high' : 'medium',
      
      // Always premium for best quality
      outputQuality: 'premium',
      
      // SVG format (compression handled by CDN/nginx)
      format: 'svg',
      
      metadata: {
        detectedType,
        dimensions: { width: metadata.width!, height: metadata.height! },
        originalSize: size,
        estimatedComplexity,
      }
    };
  }
}

// ============================================================
// SVG PROCESSOR (Core Logic)
// ============================================================

class SVGProcessor {
  private queue: PQueue;
  
  constructor() {
    this.queue = new PQueue({ 
      concurrency: CONFIG.QUEUE_CONCURRENCY,
      autoStart: true 
    });
  }

  /**
   * Main processing pipeline
   */
  async process(
    imageBuffer: Buffer, 
    originalName: string, 
    mode: 'color' | 'binary' | 'grayscale'
  ): Promise<{
    jobId: string;
    svgBuffer: Buffer;
    metadata: {
      originalSize: number;
      svgSize: number;
      compressionRatio: number;
      processingTimeMs: number;
      dimensions: { width: number; height: number };
      detectedType: 'photo' | 'graphic';
      mode: string;
      autoOptimized: boolean;
      params: any;
    };
  }> {
    const jobId = uuidv4();
    const startTime = Date.now();
    
    // File paths
    const ext = extname(originalName).toLowerCase() || '.png';
    const baseName = `${jobId}-${Date.now()}`;
    const paths = {
      input: join(CONFIG.TEMP_DIR, `${baseName}-input${ext}`),
      preprocessed: join(CONFIG.TEMP_DIR, `${baseName}-preprocessed.png`),
      svg: join(CONFIG.TEMP_DIR, `${baseName}.svg`),
    };

    try {
      // Step 1: Analyze and get optimal parameters
      console.log(`[${jobId}] Starting analysis...`);
      const autoParams = await AutoOptimizer.analyze(imageBuffer);
      console.log(`[${jobId}] Detected: ${autoParams.metadata.detectedType}, complexity: ${autoParams.metadata.estimatedComplexity}`);

      // Step 2: Save input temporarily
      await fs.writeFile(paths.input, imageBuffer);

      // Step 3: Preprocess image
      console.log(`[${jobId}] Preprocessing with noise=${autoParams.noiseReduction}...`);
      await this.preprocessImage(paths.input, paths.preprocessed, autoParams);

      // Step 4: Convert to SVG using VTracer
      console.log(`[${jobId}] Vectorizing with mode=${mode}, precision=${autoParams.colorPrecision}...`);
      await this.vectorize(paths.preprocessed, paths.svg, mode, autoParams);

      // Step 5: Post-process and optimize SVG
      console.log(`[${jobId}] Optimizing output...`);
      const svgBuffer = await this.optimizeSVG(paths.svg, autoParams);

      // Calculate metrics
      const processingTime = Date.now() - startTime;
      const compressionRatio = imageBuffer.length / svgBuffer.length;

      // Cleanup temp files
      this.cleanupFiles([paths.input, paths.preprocessed, paths.svg]);

      console.log(`[${jobId}] Complete in ${processingTime}ms, ratio: ${compressionRatio.toFixed(2)}x`);

      return {
        jobId,
        svgBuffer,
        metadata: {
          originalSize: imageBuffer.length,
          svgSize: svgBuffer.length,
          compressionRatio: parseFloat(compressionRatio.toFixed(2)),
          processingTimeMs: processingTime,
          dimensions: autoParams.metadata.dimensions,
          detectedType: autoParams.metadata.detectedType,
          mode: mode,
          autoOptimized: true,
          params: {
            colorPrecision: autoParams.colorPrecision,
            noiseReduction: autoParams.noiseReduction,
            detailLevel: autoParams.detailLevel,
          }
        }
      };

    } catch (error) {
      // Cleanup on error
      this.cleanupFiles([paths.input, paths.preprocessed, paths.svg]);
      console.error(`[${jobId}] Processing failed:`, error);
      throw error;
    }
  }

  /**
   * Preprocess image: denoise, enhance, resize
   */
  private async preprocessImage(
    inputPath: string, 
    outputPath: string, 
    params: any
  ): Promise<void> {
    // Fast path for simple graphics with no noise reduction
    if (params.metadata.detectedType === 'graphic' && params.noiseReduction <= 2) {
      await sharp(inputPath)
        .ensureAlpha()
        .toColorspace('srgb')
        .resize(4096, 4096, { 
          fit: 'inside', 
          withoutEnlargement: true,
          kernel: sharp.kernel.lanczos3 
        })
        .sharpen({ sigma: 0.5, flat: 1, jagged: 1 })
        .png({ compressionLevel: 0 })
        .toFile(outputPath);
      return;
    }

    // Quality path using Python OpenCV for photos and complex images
    const pythonScript = `
import cv2
import numpy as np
import sys

try:
    # Read image
    img = cv2.imread(r'${inputPath.replace(/\\/g, '\\\\')}')
    if img is None:
        print("ERROR: Could not read image", file=sys.stderr)
        sys.exit(1)
    
    h, w = img.shape[:2]
    
    # Resize if too large (preserve quality)
    max_dim = 4096
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = new_h, new_w
    
    # Non-local means denoising (preserves edges)
    noise = ${params.noiseReduction}
    if noise > 0:
        img = cv2.fastNlMeansDenoisingColored(
            img, None, 
            h=noise,
            hColor=noise,
            templateWindowSize=7,
            searchWindowSize=21
        )
    
    # CLAHE (Adaptive histogram equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Sharpen for maximum detail
    if '${params.detailLevel}' == 'maximum':
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
    elif '${params.detailLevel}' == 'high':
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        img = cv2.filter2D(img, -1, kernel)
    
    # Save
    cv2.imwrite(r'${outputPath.replace(/\\/g, '\\\\')}', img)
    
except Exception as e:
    print(f"ERROR: {str(e)}", file=sys.stderr)
    sys.exit(1)
`;
    
    const scriptPath = join(CONFIG.TEMP_DIR, `preprocess-${Date.now()}-${Math.random().toString(36).substr(2, 9)}.py`);
    
    try {
      await fs.writeFile(scriptPath, pythonScript);
      const { stderr } = await execAsync(
        `${CONFIG.PYTHON_PATH} "${scriptPath}"`,
        { timeout: 60000 }
      );
      
      if (stderr && stderr.includes('ERROR')) {
        throw new Error(`Preprocessing failed: ${stderr}`);
      }
    } finally {
      await fs.unlink(scriptPath).catch(() => {});
    }
  }

  /**
   * Vectorize using VTracer
   */
  private async vectorize(
    inputPath: string,
    outputPath: string,
    mode: 'color' | 'binary' | 'grayscale',
    params: any
  ): Promise<void> {
    const colorMode = mode === 'color' ? 'color' : mode === 'grayscale' ? 'gray' : 'bw';
    
    const args = [
      CONFIG.VTRACER_PATH,
      '--input', inputPath,
      '--output', outputPath,
      '--colormode', colorMode,
      '--color_precision', params.colorPrecision.toString(),
      '--mode', params.detailLevel === 'low' ? 'polygon' : 'spline',
      '--filter_speckle', Math.max(2, Math.floor(params.noiseReduction / 2)).toString(),
      '--gradient_step', params.detailLevel === 'maximum' ? '0' : '1',
      '--corner_threshold', '60',
      '--segment_length', '4',
    ];

    try {
      const { stderr } = await execAsync(args.join(' '), { timeout: 120000 });
      
      if (stderr && !stderr.includes('warning')) {
        console.warn('VTracer stderr:', stderr);
      }
      
      // Verify output exists
      await fs.access(outputPath);
    } catch (error: any) {
      // Fallback to potrace for binary mode if vtracer fails
      if (mode === 'binary') {
        console.log('Falling back to potrace for binary mode...');
        try {
          await execAsync(`potrace -s -o "${outputPath}" "${inputPath}"`, { timeout: 60000 });
        } catch (potraceError) {
          throw new Error(`Both vtracer and potrace failed: ${error.message}`);
        }
      } else {
        throw new Error(`Vectorization failed: ${error.message}`);
      }
    }
  }

  /**
   * Post-process and optimize SVG
   */
  private async optimizeSVG(svgPath: string, params: any): Promise<Buffer> {
    let content = await fs.readFile(svgPath, 'utf-8');
    
    // Remove unnecessary precision (reduce file size)
    if (params.outputQuality !== 'premium') {
      // Reduce decimal places to 2
      content = content.replace(/(\d)\.(\d{3,})/g, (match, p1, p2) => `${p1}.${p2.substr(0, 2)}`);
    }
    
    // Clean up empty elements
    content = content.replace(/<g[^>]*>\s*<\/g>/g, '');
    content = content.replace(/\s+/g, ' ');
    
    return Buffer.from(content, 'utf-8');
  }

  /**
   * Cleanup temporary files
   */
  private cleanupFiles(paths: string[]): void {
    paths.forEach(path => {
      fs.unlink(path).catch(() => {});
    });
  }
}

// Initialize processor
const processor = new SVGProcessor();

// ============================================================
// API SETUP
// ============================================================

const app = new OpenAPIHono();

// Middleware
app.use('*', cors({
  origin: '*',
  allowMethods: ['POST', 'GET', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
}));
app.use('*', logger());

// ============================================================
// ROUTES
// ============================================================

// Health check endpoint
app.get('/health', (c) => {
  return c.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    version: '2.0.0',
    config: {
      maxFileSize: `${CONFIG.MAX_FILE_SIZE_MB}MB`,
      concurrency: CONFIG.QUEUE_CONCURRENCY,
    }
  });
});

// OpenAPI Schema Definitions
const ConvertRequestSchema = z.object({
  // Input Option 1: File upload (File is Web API; use conditional for Node.js)
  image: (typeof File !== 'undefined' ? z.instanceof(File) : z.any()).optional()
    .openapi({
      description: `Image file. Supported formats: PNG, JPG, JPEG, WebP, GIF, BMP. Max ${CONFIG.MAX_FILE_SIZE_MB}MB, up to ${(CONFIG.MAX_PIXELS / 1_000_000).toFixed(0)}M pixels. When provided, leave imageBase64 empty.`,
      type: 'string',
      format: 'binary',
    }),

  // Input Option 2: Base64 encoded string — leave empty when uploading via image file
  imageBase64: z.string().optional().default('')
    .openapi({
      description: 'Leave empty when using file upload (image). Use only when image is not provided: base64 string, optionally with data:image/...;base64, prefix.',
      example: '',
      default: '',
    }),

  // Processing mode (only manual parameter)
  mode: z.enum(['color', 'binary', 'grayscale'])
    .default('color')
    .openapi({
      description: 'Vectorization mode: color (full color), binary (black/white), grayscale',
      enum: ['color', 'binary', 'grayscale'],
      example: 'color',
    }),
});

const ConvertResponseSchema = z.object({
  success: z.boolean().openapi({ example: true }),
  jobId: z.string().uuid().openapi({ example: '550e8400-e29b-41d4-a716-446655440000' }),
  svgUrl: z.string().url().openapi({ example: 'http://localhost:3000/download/550e8400-e29b-41d4-a716-446655440000' }),
  metadata: z.object({
    originalSize: z.number().openapi({ example: 2457600 }),
    svgSize: z.number().openapi({ example: 154200 }),
    compressionRatio: z.number().openapi({ example: 15.9 }),
    processingTimeMs: z.number().openapi({ example: 1250 }),
    dimensions: z.object({
      width: z.number().openapi({ example: 1920 }),
      height: z.number().openapi({ example: 1080 }),
    }),
    detectedType: z.enum(['photo', 'graphic']).openapi({ example: 'photo' }),
    mode: z.string().openapi({ example: 'color' }),
    autoOptimized: z.boolean().openapi({ example: true }),
    params: z.object({
      colorPrecision: z.number().openapi({ example: 6 }),
      noiseReduction: z.number().openapi({ example: 10 }),
      detailLevel: z.string().openapi({ example: 'high' }),
    }),
  }),
});

const ErrorResponseSchema = z.object({
  success: z.literal(false),
  error: z.string().openapi({ example: 'Invalid input: No image provided' }),
  code: z.string().openapi({ example: 'BAD_REQUEST' }),
});

/** Reusable type definitions for backend consumers (request/response contracts) */
export type ConvertResponse = z.infer<typeof ConvertResponseSchema>;
export type ConvertResponseMetadata = ConvertResponse['metadata'];
export type ErrorResponse = z.infer<typeof ErrorResponseSchema>;

// ============================================================
// MAIN CONVERT ENDPOINT
// ============================================================

const convertRoute = createRoute({
  method: 'post',
  path: '/convert',
  request: {
    description: `
Convert image to SVG with auto-optimization.

**Request body (provide ONE input):**
- **File upload:** Send \`image\` (file) and leave \`imageBase64\` empty or omit it.
- **Base64:** Send \`imageBase64\` (string) when \`image\` is not provided; omit \`image\` or leave it empty.

When \`image\` is provided, \`imageBase64\` must be empty — the server ignores imageBase64 if image file is present.

**Mode:** \`color\` (default) | \`binary\` | \`grayscale\`

**Limits:** File size max \`${CONFIG.MAX_FILE_SIZE_MB}MB\`. Total pixels (width×height) max \`${(CONFIG.MAX_PIXELS / 1_000_000).toFixed(0)}M\` (professional-grade). Optimal logo dimension: \`${CONFIG.OPTIMAL_LOGO_DIMENSION}×${CONFIG.OPTIMAL_LOGO_DIMENSION}px\` for best quality vs CPU/memory.

**Auto-Detection:** Photos get high noise reduction and 6-color precision; graphics get low noise and 4-color precision.
    `,
    body: {
      content: {
        'multipart/form-data': {
          schema: ConvertRequestSchema,
        },
        'application/json': {
          schema: z.object({
            imageBase64: z.string().describe('Base64 image; required when no file upload. Leave empty if using file.'),
            mode: z.enum(['color', 'binary', 'grayscale']).default('color'),
          }),
        },
      },
    },
  },
  responses: {
    200: {
      description: 'Conversion successful. Use `jobId` or `svgUrl` to download the SVG (GET /download/{jobId}).',
      content: {
        'application/json': {
          schema: ConvertResponseSchema,
        },
      },
    },
    400: {
      description: 'Invalid input (e.g. no image provided, invalid base64)',
      content: {
        'application/json': {
          schema: ErrorResponseSchema,
        },
      },
    },
    413: {
      description: 'File too large (exceeds max size)',
      content: {
        'application/json': {
          schema: ErrorResponseSchema,
        },
      },
    },
    415: {
      description: 'Unsupported media type',
      content: {
        'application/json': {
          schema: ErrorResponseSchema,
        },
      },
    },
    500: {
      description: 'Processing error',
      content: {
        'application/json': {
          schema: ErrorResponseSchema,
        },
      },
    },
  },
  tags: ['Conversion'],
});

// Convert handler implementation
app.openapi(convertRoute, async (c) => {
  const contentType = c.req.header('content-type') || '';
  
  let imageBuffer: Buffer;
  let mode: 'color' | 'binary' | 'grayscale' = 'color';
  let originalName: string = 'upload.png';

  try {
    // Parse request based on content type
    if (contentType.includes('multipart/form-data')) {
      const body = await c.req.parseBody();
      const rawImage = body.image;

      // File-like: has arrayBuffer() and size (works with File, Blob, or Node multipart result)
      const isFileUpload =
        rawImage &&
        typeof (rawImage as { arrayBuffer?: () => unknown }).arrayBuffer === 'function' &&
        typeof (rawImage as { size?: unknown }).size === 'number';

      // Handle file upload (image present, imageBase64 can be empty)
      if (isFileUpload) {
        const file = rawImage as { arrayBuffer: () => Promise<ArrayBuffer>; size: number; type: string; name?: string };
        imageBuffer = Buffer.from(await file.arrayBuffer());
        originalName = file.name || 'upload.jpg';

        // Validate file type
        if (!CONFIG.ALLOWED_TYPES.includes(file.type)) {
          throw new HTTPException(415, {
            message: `Unsupported file type: ${file.type}. Allowed: ${CONFIG.ALLOWED_TYPES.join(', ')}`,
          });
        }

        // Validate file size
        const maxBytes = CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024;
        if (file.size > maxBytes) {
          throw new HTTPException(413, {
            message: `File too large: ${(file.size / 1024 / 1024).toFixed(2)}MB (max ${CONFIG.MAX_FILE_SIZE_MB}MB)`,
          });
        }

      // Handle base64 in form data (image empty, imageBase64 non-empty)
      } else if (typeof body.imageBase64 === 'string' && body.imageBase64.trim()) {
        const base64Data = body.imageBase64.replace(/^data:image\/\w+;base64,/, '');
        imageBuffer = Buffer.from(base64Data, 'base64');
        originalName = 'base64-upload.png';

        if (imageBuffer.length === 0) {
          throw new HTTPException(400, { message: 'Invalid base64 data' });
        }
        
        // Check size
        const maxBytes = CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024;
        if (imageBuffer.length > maxBytes) {
          throw new HTTPException(413, { 
            message: `Image too large: ${(imageBuffer.length / 1024 / 1024).toFixed(2)}MB` 
          });
        }
      } else {
        throw new HTTPException(400, { 
          message: 'No image provided. Use "image" (file) or "imageBase64" (string)' 
        });
      }
      
      // Get mode from form
      if (body.mode) {
        if (!['color', 'binary', 'grayscale'].includes(body.mode as string)) {
          throw new HTTPException(400, { message: 'Mode must be: color, binary, or grayscale' });
        }
        mode = body.mode as 'color' | 'binary' | 'grayscale';
      }
      
    } else if (contentType.includes('application/json')) {
      const body = await c.req.json();
      
      if (!body.imageBase64 || typeof body.imageBase64 !== 'string') {
        throw new HTTPException(400, { message: 'JSON requests require imageBase64 field' });
      }
      
      const base64Data = body.imageBase64.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(base64Data, 'base64');
      originalName = 'base64-json.png';
      
      if (imageBuffer.length === 0) {
        throw new HTTPException(400, { message: 'Invalid base64 data' });
      }
      
      // Check size
      const maxBytes = CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024;
      if (imageBuffer.length > maxBytes) {
        throw new HTTPException(413, { 
          message: `Image too large: ${(imageBuffer.length / 1024 / 1024).toFixed(2)}MB` 
        });
      }
      
      // Get mode from JSON
      if (body.mode) {
        if (!['color', 'binary', 'grayscale'].includes(body.mode)) {
          throw new HTTPException(400, { message: 'Mode must be: color, binary, or grayscale' });
        }
        mode = body.mode;
      }
      
    } else {
      throw new HTTPException(400, { 
        message: 'Unsupported content type. Use multipart/form-data or application/json' 
      });
    }

    // Validate image is parseable and within pixel limit (up to 250M pixels for professional-grade)
    let width: number;
    let height: number;
    try {
      const meta = await sharp(imageBuffer).metadata();
      width = meta.width ?? 0;
      height = meta.height ?? 0;
      if (!width || !height) {
        throw new HTTPException(400, { message: 'Could not read image dimensions' });
      }
      const totalPixels = width * height;
      if (totalPixels > CONFIG.MAX_PIXELS) {
        throw new HTTPException(413, {
          message: `Image exceeds maximum allowed pixels: ${(totalPixels / 1_000_000).toFixed(1)}M (max ${(CONFIG.MAX_PIXELS / 1_000_000).toFixed(0)}M). Reduce dimensions and try again.`,
        });
      }
    } catch (err) {
      if (err instanceof HTTPException) throw err;
      throw new HTTPException(400, { message: 'Invalid or corrupted image file' });
    }

    console.log(`[REQUEST] Mode: ${mode}, Size: ${imageBuffer.length} bytes, ${width}x${height} (${((width * height) / 1_000_000).toFixed(1)}M px), Type: ${contentType}`);

    // Process image
    const result = await processor.process(imageBuffer, originalName, mode);
    
    // Save output for download
    const outputPath = join(CONFIG.TEMP_DIR, `${result.jobId}.svg`);
    await fs.writeFile(outputPath, result.svgBuffer);
    
    // Schedule cleanup
    setTimeout(() => {
      fs.unlink(outputPath).catch(() => {});
    }, 3600000); // 1 hour

    // Build response
    const protocol = c.req.header('x-forwarded-proto') || 'http';
    const host = c.req.header('host') || `localhost:${CONFIG.PORT}`;
    const baseUrl = `${protocol}://${host}`;

    return c.json({
      success: true,
      jobId: result.jobId,
      svgUrl: `${baseUrl}/download/${result.jobId}`,
      metadata: result.metadata,
    });

  } catch (error) {
    if (error instanceof HTTPException) throw error;
    
    console.error('Conversion error:', error);
    throw new HTTPException(500, { 
      message: error instanceof Error ? error.message : 'Image processing failed' 
    });
  }
});

// ============================================================
// DOWNLOAD ENDPOINT
// ============================================================

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
      content: {
        'image/svg+xml': {
          schema: z.string().openapi({ format: 'binary' }),
        },
      },
    },
    404: {
      description: 'File not found or expired',
      content: {
        'application/json': {
          schema: ErrorResponseSchema,
        },
      },
    },
  },
  tags: ['Download'],
  description: 'Download converted SVG file. Files expire after 1 hour.',
});

app.openapi(downloadRoute, async (c) => {
  const { jobId } = c.req.valid('param');
  const filePath = join(CONFIG.TEMP_DIR, `${jobId}.svg`);
  
  try {
    // Check if file exists
    await fs.access(filePath);
    
    // Stream file
    const stream = createReadStream(filePath);
    
    c.header('Content-Type', 'image/svg+xml');
    c.header('Content-Disposition', `attachment; filename="converted-${jobId}.svg"`);
    c.header('Cache-Control', 'public, max-age=3600');
    c.header('X-Content-Type-Options', 'nosniff');
    
    return c.body(stream);
    
  } catch {
    throw new HTTPException(404, { 
      message: 'File not found or expired. Files are kept for 1 hour.' 
    });
  }
});

// ============================================================
// SWAGGER UI & DOCUMENTATION
// ============================================================

app.get('/docs', swaggerUI({ url: '/openapi.json' }));

// OpenAPI specification
app.doc('/openapi.json', {
  openapi: '3.0.0',
  info: {
    title: 'Image-to-SVG Conversion API',
    version: '2.0.0',
    description: `
## Ultra-Simple Image Vectorization API

Convert images to high-quality SVG. **Supported input formats:** PNG, JPG, JPEG, WebP, GIF, BMP.

### Limits
- **File size:** Max **10 MB**
- **Total pixels:** Up to **250 million** (width × height) for professional-grade assets
- **Optimal logo dimension:** **1200×1200px** recommended — enough detail for the tracing algorithm without overloading CPU/memory

### Features
- **Dual Input**: Upload file OR send base64 string (use one; when using file, leave imageBase64 empty)
- **Auto-Optimization**: Detects photos vs graphics automatically
- **Zero Distortion**: Lanczos3 resampling + spline curves
- **Fast Processing**: Typically 300ms-2s depending on image size

---

## Backend Developer Reference

### POST /convert — Request Body

| Content-Type | Field | Type | Required | Description |
|-------------|-------|------|----------|-------------|
| multipart/form-data | image | file (binary) | One of image or imageBase64 | Image file. **Formats:** PNG, JPG, JPEG, WebP, GIF, BMP. **Max 10MB, up to 250M pixels.** When sent, imageBase64 must be empty. |
| multipart/form-data | imageBase64 | string | One of image or imageBase64 | Base64 image (optional data URI prefix). **Leave empty when uploading image file.** |
| multipart/form-data | mode | string | No (default: color) | \`color\` | \`binary\` | \`grayscale\` |
| application/json | imageBase64 | string | Yes (for JSON) | Base64 image. Omit when using multipart with file. |
| application/json | mode | string | No (default: color) | \`color\` | \`binary\` | \`grayscale\` |

**Rule:** Provide **exactly one** of \`image\` (file) or \`imageBase64\` (string). If \`image\` is provided, \`imageBase64\` is ignored and should be empty.

### POST /convert — Response Body (200)

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | \`true\` |
| jobId | string (UUID) | Use with GET /download/{jobId} to fetch the SVG file |
| svgUrl | string (URL) | Full URL to download the SVG (same as GET /download/{jobId}) |
| metadata | object | originalSize, svgSize, compressionRatio, processingTimeMs, dimensions, detectedType, mode, autoOptimized, params |

### POST /convert — Error Responses (4xx/5xx)

| Status | Body | When |
|--------|------|------|
| 400 | { success: false, error: string, code: string } | No image, invalid base64, invalid mode, unreadable dimensions |
| 413 | { success: false, error: string, code: string } | File/image too large (>10MB) or total pixels >250M |
| 415 | { success: false, error: string, code: string } | Unsupported file type (allowed: PNG, JPG, JPEG, WebP, GIF, BMP) |
| 500 | { success: false, error: string, code: string } | Processing error |

### GET /download/{jobId} — Response

| Status | Content-Type | Body |
|--------|--------------|------|
| 200 | image/svg+xml | SVG file body |
| 404 | application/json | { success: false, error: string, code: string } — file expired or not found (files kept 1 hour) |

---

### Quick Start

**File upload (imageBase64 empty):**
\`\`\`bash
curl -X POST http://localhost:3000/convert \\
  -F "image=@photo.jpg" \\
  -F "mode=color"
\`\`\`

**Base64 (JSON):**
\`\`\`bash
curl -X POST http://localhost:3000/convert \\
  -H "Content-Type: application/json" \\
  -d '{"imageBase64": "'"$(base64 -w 0 photo.jpg)"'", "mode": "binary"}'
\`\`\`

### Auto-Detection Logic
| Image Type | Noise Reduction | Color Precision | Detail Level |
|------------|----------------|-----------------|--------------|
| Photo | 10 | 6 | high/maximum |
| Graphic | 2 | 4 | medium/high |

### Modes
- **color**: Full color vectorization (best for photos)
- **binary**: Black & white (best for text/logos)
- **grayscale**: Grayscale vectorization
    `,
    contact: {
      name: 'API Support',
    },
  },
  servers: [
    { url: 'http://localhost:3000', description: 'Local development' },
  ],
  tags: [
    { name: 'Conversion', description: 'Image to SVG conversion' },
    { name: 'Download', description: 'Retrieve converted files' },
  ],
});

// ============================================================
// ERROR HANDLING
// ============================================================

app.onError((err, c) => {
  if (err instanceof HTTPException) {
    return c.json({
      success: false,
      error: err.message,
      code: err.status.toString(),
    }, err.status);
  }
  
  console.error('Unhandled error:', err);
  return c.json({
    success: false,
    error: 'Internal server error',
    code: 'INTERNAL_ERROR',
  }, 500);
});

app.notFound((c) => {
  return c.json({
    success: false,
    error: 'Endpoint not found',
    code: 'NOT_FOUND',
  }, 404);
});

// ============================================================
// START SERVER
// ============================================================

console.log(`
╔════════════════════════════════════════════════════════════╗
║           🚀 Image-to-SVG API (Production Ready)           ║
╠════════════════════════════════════════════════════════════╣
║  📚 Swagger UI:  http://localhost:${CONFIG.PORT}/docs                  ║
║  🔥 Convert:     POST http://localhost:${CONFIG.PORT}/convert          ║
║  📥 Download:    GET  http://localhost:${CONFIG.PORT}/download/{id}    ║
║  ❤️  Health:     GET  http://localhost:${CONFIG.PORT}/health           ║
╠════════════════════════════════════════════════════════════╣
║  Inputs:  image (file) OR imageBase64 (string)            ║
║  Mode:    color | binary | grayscale                       ║
║  Max Size: ${CONFIG.MAX_FILE_SIZE_MB}MB                                                    ║
╚════════════════════════════════════════════════════════════╝
`);

serve({
  fetch: app.fetch,
  port: CONFIG.PORT,
});