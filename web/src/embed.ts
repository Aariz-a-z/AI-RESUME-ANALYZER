/**
 * Sentence embeddings in the browser.
 *
 * Same weights as the Python version (all-MiniLM-L6-v2), quantised to int8 so
 * the download is ~23MB instead of ~90MB. Mean pooling + L2 normalisation makes
 * these directly comparable to SentenceTransformer's
 * `encode(..., normalize_embeddings=True)`, so a dot product is the cosine.
 */

import { pipeline, env, type FeatureExtractionPipeline } from '@huggingface/transformers';
import type { Embedder } from './analyzer.ts';

// There are no local model files to serve; always fetch from the Hub CDN.
env.allowLocalModels = false;

export const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';

export interface LoadProgress {
  status: string;
  file?: string;
  progress?: number;
}

let pipePromise: Promise<FeatureExtractionPipeline> | null = null;

export function loadEmbedder(onProgress?: (p: LoadProgress) => void): Promise<FeatureExtractionPipeline> {
  pipePromise ??= pipeline('feature-extraction', MODEL_ID, {
    dtype: 'q8',
    progress_callback: onProgress as never,
  });
  return pipePromise;
}

/** Batched so a long resume does not build one enormous tensor. */
export async function createEmbedder(
  onProgress?: (p: LoadProgress) => void,
  batchSize = 16
): Promise<Embedder> {
  const extractor = await loadEmbedder(onProgress);

  return async (texts: string[]): Promise<number[][]> => {
    const vectors: number[][] = [];
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const output = await extractor(batch, { pooling: 'mean', normalize: true });
      vectors.push(...(output.tolist() as number[][]));
    }
    return vectors;
  };
}
