/**
 * Engine settings. The defaults are the Python engine's (`khmerocr.config`),
 * so both engines flag the same images and fields for review.
 */
export interface QualityThresholds {
  /** Images whose shorter side is below this many pixels are rejected. */
  minShortSide: number;
  /** Decoded pixel budget; protects against decompression bombs. */
  maxPixels: number;
  /** Variance of the Laplacian measured at a 1000px long side. */
  blurWarn: number;
  blurReject: number;
  /** Fraction of fully saturated (clipped white) pixels. */
  glareWarnRatio: number;
  glareRejectRatio: number;
  /** Mean grey level bounds (0-255). */
  minBrightness: number;
  maxBrightness: number;
  /** Grey-level standard deviation below which the image is considered blank. */
  minContrast: number;
}

export interface EngineConfig {
  maxImageBytes: number;
  quality: QualityThresholds;
  /** Fields whose recognition confidence falls below this are flagged for review. */
  reviewConfidenceThreshold: number;
  /** Raise IMAGE_QUALITY_TOO_LOW for blocking quality issues instead of extracting anyway. */
  rejectLowQuality: boolean;
  /** Retry with the card turned 180 degrees when the first pass needs review. */
  tryAlternateOrientations: boolean;
  /** Relative padding added around each template region before recognition. */
  regionPadding: number;
}

export interface EngineConfigOverrides extends Partial<Omit<EngineConfig, "quality">> {
  quality?: Partial<QualityThresholds>;
}

export const DEFAULT_QUALITY: Readonly<QualityThresholds> = Object.freeze({
  minShortSide: 300,
  maxPixels: 40_000_000,
  blurWarn: 120.0,
  blurReject: 8.0,
  glareWarnRatio: 0.015,
  glareRejectRatio: 0.12,
  minBrightness: 35.0,
  maxBrightness: 240.0,
  minContrast: 12.0,
});

export function resolveConfig(overrides: EngineConfigOverrides = {}): EngineConfig {
  const { quality, ...rest } = overrides;
  return {
    maxImageBytes: 10 * 1024 * 1024,
    reviewConfidenceThreshold: 0.85,
    rejectLowQuality: true,
    tryAlternateOrientations: true,
    regionPadding: 0.01,
    ...withoutUndefined(rest),
    quality: { ...DEFAULT_QUALITY, ...withoutUndefined(quality ?? {}) },
  };
}

function withoutUndefined<T extends object>(value: T): Partial<T> {
  return Object.fromEntries(Object.entries(value).filter(([, v]) => v !== undefined)) as Partial<T>;
}
