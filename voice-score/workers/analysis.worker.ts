import type { IpadicFeatures, Tokenizer } from "kuromoji";
import { PhonemizerJa } from "charsiu-js/core";
import {
  analyzePitch,
  hzToMidi,
  mergePitchFrames,
  preparePitchAudio,
  splitJapaneseMoras,
  type PitchFrame,
  type Segment,
  type WordReading,
} from "../lib/analysis";
import { normalizeJapaneseTokens, type PronounceableJapaneseToken } from "../lib/japanese-reading";
import { ctcEditAlign } from "../lib/ctc-edit-align";

type AnalysisRequest = {
  type: "analyze";
  audio: Float32Array;
  text: string;
};

type Ort = typeof import("onnxruntime-web/wasm");

type AnalysisRuntime = {
  ort: Ort;
  session: import("onnxruntime-web/wasm").InferenceSession;
};

type WorkerEndpoint = {
  onmessage: ((event: MessageEvent<AnalysisRequest>) => void) | null;
  postMessage: (message: unknown) => void;
};

type KuromojiNamespace = {
  builder: (options: { dicPath: string }) => {
    build: (callback: (reason: unknown, tokenizer: Tokenizer<IpadicFeatures> | undefined) => void) => void;
  };
};

const endpoint = self as unknown as WorkerEndpoint;
const ALIGN_MODEL = "https://huggingface.co/mnaoizyyy/charsiu-js-models/resolve/main/japanese-hubert-base-phoneme-ctc/model_quantized.onnx";
const KUROMOJI_DICT = "https://cdn.jsdelivr.net/npm/kuromoji@0.1.2/dict/";
const KUROMOJI_SCRIPT = "https://cdn.jsdelivr.net/npm/kuromoji@0.1.2/build/kuromoji.js";
const SWIFT_F0_MODEL = "https://huggingface.co/FredrikKarlssonSpeech/swift-f0-onnx/resolve/main/onnx/model.onnx?download=true";
const SAMPLE_RATE = 16_000;
const ALIGN_CORE_SECONDS = 16;
const ALIGN_OVERLAP_SECONDS = 1.5;
const SWIFT_HOP = 256;
const SWIFT_CHUNK_SAMPLES = SAMPLE_RATE * 30;
const SWIFT_CONTEXT_SAMPLES = SWIFT_HOP * 11;
const MORA_PHONES = new Set(["a", "i", "u", "e", "o", "N", "cl"]);
// This vocabulary is only 47 entries and is part of the Charsiu Japanese
// model.  Keep it in the worker so analysis does not depend on a second CDN
// request that could leave the UI stuck at 48% when that request stalls.
const PHONE_VOCAB_JA: Record<string, number> = {
  PAD: 0, UNK: 1, SOS: 2, EOS: 3,
  a: 4, i: 5, u: 6, e: 7, o: 8, I: 9, U: 10,
  k: 11, g: 12, s: 13, z: 14, t: 15, d: 16, n: 17,
  h: 18, b: 19, p: 20, m: 21, y: 22, r: 23, w: 24,
  f: 25, j: 26, v: 27, N: 28, cl: 29, sh: 30, ch: 31,
  ts: 32, ky: 33, gy: 34, hy: 35, by: 36, py: 37, my: 38,
  ny: 39, ry: 40, fy: 41, dy: 42, kw: 43, gw: 44, pau: 45, sil: 46,
};
let runtimePromise: Promise<AnalysisRuntime> | null = null;
let kuromojiTokenizerPromise: Promise<Tokenizer<IpadicFeatures>> | null = null;

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), timeoutMs);
    promise.then(
      (value) => { clearTimeout(timer); resolve(value); },
      (reason) => { clearTimeout(timer); reject(reason); },
    );
  });
}

async function fetchAlignmentModel() {
  const controller = new AbortController();
  let stalled = false;
  let stallTimer = setTimeout(() => { stalled = true; controller.abort(); }, 120_000);
  const resetStallTimer = () => {
    clearTimeout(stallTimer);
    stallTimer = setTimeout(() => { stalled = true; controller.abort(); }, 120_000);
  };

  try {
    const response = await fetch(ALIGN_MODEL, { signal: controller.signal });
    if (!response.ok) throw new Error(`音素モデルの取得に失敗しました（HTTP ${response.status}）`);
    if (!response.body) return new Uint8Array(await response.arrayBuffer());

    const expectedBytes = Number(response.headers.get("content-length")) || 123 * 1024 * 1024;
    const reader = response.body.getReader();
    const pieces: Uint8Array[] = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      resetStallTimer();
      pieces.push(value);
      received += value.byteLength;
      const downloadPercent = Math.min(99, Math.round((received / expectedBytes) * 100));
      endpoint.postMessage({
        type: "progress",
        stage: "align",
        percent: 58 + Math.round(downloadPercent * 0.1),
        message: `日本語 HuBERT 音素モデルを取得しています（${downloadPercent}%）`,
      });
    }
    const merged = new Uint8Array(received);
    let offset = 0;
    for (const piece of pieces) { merged.set(piece, offset); offset += piece.byteLength; }
    return merged;
  } catch (reason) {
    if (stalled) throw new Error("音素モデルのダウンロードが2分以上進まなかったため中止しました。通信状態を確認して再実行してください");
    throw reason;
  } finally {
    clearTimeout(stallTimer);
  }
}

function fallbackReadingForAlignedSurface(surface: string) {
  return normalizeJapaneseTokens([{ surface_form: surface }])[0]?.reading ?? surface;
}

function segmentsFromEditedAlignment(
  frameTokens: Int32Array,
  targetIds: number[],
  id2phone: Record<string, string>,
  resolution: number,
) {
  const firstFrames = new Int32Array(targetIds.length).fill(-1);
  const lastFrames = new Int32Array(targetIds.length).fill(-1);
  frameTokens.forEach((target, frame) => {
    if (target < 0) return;
    if (firstFrames[target] < 0) firstFrames[target] = frame;
    lastFrames[target] = frame;
  });

  const segments: Segment[] = [];
  let cursor = 0;
  for (let target = 0; target < targetIds.length; target += 1) {
    if (firstFrames[target] < 0) continue;
    const start = firstFrames[target] * resolution;
    const end = (lastFrames[target] + 1) * resolution;
    if (start > cursor + resolution * 0.5) segments.push([cursor, start, "[SIL]"]);
    segments.push([start, end, id2phone[String(targetIds[target])] ?? ""]);
    cursor = end;
  }
  const duration = frameTokens.length * resolution;
  if (duration > cursor + resolution * 0.5) segments.push([cursor, duration, "[SIL]"]);
  return segments;
}

async function alignWholeRecording(
  session: import("onnxruntime-web/wasm").InferenceSession,
  ort: typeof import("onnxruntime-web/wasm"),
  phonemizer: PhonemizerJa,
  audio: Float32Array,
  text: string,
) {
  const duration = audio.length / SAMPLE_RATE;
  const coreCount = Math.max(1, Math.ceil(duration / ALIGN_CORE_SECONDS));
  const logitParts: Float32Array[] = [];
  let totalFrames = 0;
  let vocabSize = 0;

  for (let index = 0; index < coreCount; index += 1) {
    const coreStart = index * ALIGN_CORE_SECONDS;
    const coreEnd = Math.min(duration, (index + 1) * ALIGN_CORE_SECONDS);
    const inputStart = Math.max(0, coreStart - ALIGN_OVERLAP_SECONDS);
    const inputEnd = Math.min(duration, coreEnd + ALIGN_OVERLAP_SECONDS);
    endpoint.postMessage({
      type: "progress",
      stage: "align",
      percent: 70 + Math.round(((index + 1) / coreCount) * 11),
      message: `音声全体の音素特徴を抽出しています（${index + 1}/${coreCount}区間）`,
    });

    const waveform = audio.slice(Math.floor(inputStart * SAMPLE_RATE), Math.ceil(inputEnd * SAMPLE_RATE));
    const input = new ort.Tensor("float32", waveform, [1, waveform.length]);
    try {
      const outputs = await withTimeout(
        session.run({ input_values: input }),
        300_000,
        `音素特徴の抽出が5分以内に完了しませんでした（${index + 1}/${coreCount}区間）`,
      );
      const logits = outputs.logits;
      if (!logits || logits.dims.length < 3) throw new Error("音素モデルから特徴量を取得できませんでした");
      try {
        const frames = Number(logits.dims[1]);
        const vocab = Number(logits.dims[2]);
        if (vocabSize && vocabSize !== vocab) throw new Error("音素モデルの出力形式が区間ごとに一致しません");
        vocabSize = vocab;
        const data = logits.data as Float32Array;
        const secondsPerFrame = (inputEnd - inputStart) / Math.max(1, frames);
        const keptRows: number[] = [];
        for (let frame = 0; frame < frames; frame += 1) {
          const center = inputStart + (frame + 0.5) * secondsPerFrame;
          const belongsToCore = center >= coreStart && (index === coreCount - 1 ? center <= coreEnd : center < coreEnd);
          if (belongsToCore) keptRows.push(frame);
        }
        const part = new Float32Array(keptRows.length * vocab);
        keptRows.forEach((frame, outputIndex) => {
          part.set(data.subarray(frame * vocab, (frame + 1) * vocab), outputIndex * vocab);
        });
        logitParts.push(part);
        totalFrames += keptRows.length;
      } finally {
        logits.dispose();
      }
    } finally {
      input.dispose();
    }
  }

  const { words, targetIds, groupLens } = phonemizer.phonemize(text);
  if (!targetIds.length) throw new Error("歌詞から発音可能な日本語を取得できませんでした");

  endpoint.postMessage({ type: "progress", stage: "align", percent: 83, message: "欠落を許容しながら歌詞全文を音声へ整列しています" });
  const logits = new Float32Array(totalFrames * vocabSize);
  let offset = 0;
  for (const part of logitParts) { logits.set(part, offset); offset += part.length; }
  const alignment = ctcEditAlign(logits, totalFrames, vocabSize, targetIds, phonemizer.blankIdx, phonemizer.id2phone);
  const frameTokens = alignment.frameTokens.slice();
  for (let frame = 0; frame < frameTokens.length; frame += 1) {
    const target = frameTokens[frame];
    if (target >= 0 && !alignment.supportedTargets[target]) frameTokens[frame] = -1;
  }
  const resolution = duration / totalFrames;
  return {
    phones: segmentsFromEditedAlignment(frameTokens, targetIds, phonemizer.id2phone, resolution),
    phoneIds: targetIds,
    sourceWords: words,
    groupLens,
    frameTokens,
    supportedTargets: alignment.supportedTargets,
    resolution,
  };
}

async function analyzeSwiftF0(
  ort: typeof import("onnxruntime-web/wasm"),
  waveform: Float32Array,
): Promise<PitchFrame[]> {
  const response = await fetch(SWIFT_F0_MODEL);
  if (!response.ok) throw new Error(`SwiftF0モデルの読み込みに失敗しました（HTTP ${response.status}）`);
  const modelBytes = new Uint8Array(await response.arrayBuffer());
  const session = await withTimeout(
    ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] }),
    90_000,
    "SwiftF0モデルを準備できませんでした",
  );
  const enhanced = preparePitchAudio(waveform, SAMPLE_RATE);
  const frames: PitchFrame[] = [];
  const fmin = new ort.Tensor("float32", new Float32Array([55]), []);
  const fmax = new ort.Tensor("float32", new Float32Array([1400]), []);

  try {
    for (let chunkStart = 0; chunkStart < enhanced.length; chunkStart += SWIFT_CHUNK_SAMPLES) {
      const chunkEnd = Math.min(enhanced.length, chunkStart + SWIFT_CHUNK_SAMPLES);
      const paddedStart = Math.max(0, chunkStart - SWIFT_CONTEXT_SAMPLES);
      const paddedEnd = Math.min(enhanced.length, chunkEnd + SWIFT_CONTEXT_SAMPLES);
      const chunk = enhanced.slice(paddedStart, paddedEnd);
      const audioInput = new ort.Tensor("float32", chunk, [1, chunk.length]);
      let outputs: Record<string, import("onnxruntime-web/wasm").Tensor>;
      try {
        outputs = await session.run({ audio: audioInput, fmin, fmax });
      } catch {
        // The public Hugging Face export names its input/output tensors
        // input_audio and pitch_hz; the original local export used audio/pitch.
        outputs = await session.run({ input_audio: audioInput });
      }
      const pitchTensor = outputs.pitch ?? outputs.pitch_hz;
      const confidenceTensor = outputs.confidence;
      if (!pitchTensor || !confidenceTensor) throw new Error("SwiftF0の出力形式を認識できませんでした");
      const pitches = pitchTensor.data as Float32Array | Float64Array;
      const confidences = confidenceTensor.data as Float32Array;
      for (let index = 0; index < confidences.length; index += 1) {
        const sample = paddedStart + index * SWIFT_HOP;
        if (sample < chunkStart || sample >= chunkEnd) continue;
        const frequency = pitches[index];
        let sum = 0;
        const rmsEnd = Math.min(enhanced.length, sample + SWIFT_HOP);
        for (let audioIndex = sample; audioIndex < rmsEnd; audioIndex += 1) {
          sum += enhanced[audioIndex] * enhanced[audioIndex];
        }
        frames.push({
          time: sample / SAMPLE_RATE,
          frequency: Number.isFinite(frequency) && frequency >= 55 && frequency <= 1400 ? frequency : null,
          midi: Number.isFinite(frequency) && frequency >= 55 && frequency <= 1400 ? hzToMidi(frequency) : null,
          rms: Math.sqrt(sum / Math.max(1, rmsEnd - sample)),
          confidence: confidences[index],
          source: "swiftf0",
        });
      }
    }
  } finally {
    await session.release();
  }
  return frames;
}

/**
 * kuromoji's browser bundle includes zlib.js. Bundling that old CommonJS
 * file through Vite turns its top-level `this` into undefined, which causes
 * `Cannot use 'in' operator to search for 'Zlib' in undefined`. Loading the
 * published UMD bundle as a classic worker script preserves its intended
 * global scope and avoids the broken CommonJS transform.
 */
function loadKuromoji(): KuromojiNamespace {
  const scope = globalThis as typeof globalThis & {
    kuromoji?: KuromojiNamespace;
    importScripts?: (...urls: string[]) => void;
  };
  if (!scope.kuromoji) {
    if (!scope.importScripts) throw new Error("Worker が importScripts に対応していません");
    scope.importScripts(KUROMOJI_SCRIPT);
  }
  if (!scope.kuromoji) throw new Error("kuromoji を読み込めませんでした。ネットワーク接続を確認してください");
  return scope.kuromoji;
}

function canTokenizeWithoutDictionary(value: string) {
  const normalized = value.normalize("NFKC");
  // Numeric dates such as 11月31日 are handled by normalizeJapaneseTokens.
  // Any other kanji needs Kuromoji because its reading may be ambiguous.
  const withoutNumericCounters = normalized.replace(/\d+(?:月|日)/gu, "");
  return !/[々〇〆一-龯]/u.test(withoutNumericCounters);
}

function tokenizeKanaText(value: string): PronounceableJapaneseToken[] {
  const surfaces = value.normalize("NFKC").match(
    /[ぁ-ゖァ-ヶー]+|[+-]?\d+(?:\.\d+)?|[月日]/gu,
  ) ?? [];
  return normalizeJapaneseTokens(surfaces.map((surface_form) => ({ surface_form })));
}

async function getTokenizeForAlignment(value: string) {
  if (canTokenizeWithoutDictionary(value)) {
    endpoint.postMessage({
      type: "progress",
      stage: "align",
      percent: 50,
      message: "かな歌詞の読みを準備しました（日本語辞書は不要です）",
    });
    return tokenizeKanaText;
  }

  endpoint.postMessage({
    type: "progress",
    stage: "align",
    percent: 48,
    message: "漢字を含むため日本語辞書を準備しています",
  });
  if (!kuromojiTokenizerPromise) {
    kuromojiTokenizerPromise = (async () => {
      const kuromoji = loadKuromoji();
      endpoint.postMessage({ type: "progress", stage: "align", percent: 52, message: "日本語の読み辞書を読み込んでいます" });
      return withTimeout(new Promise<Tokenizer<IpadicFeatures>>((resolve, reject) => {
        kuromoji.builder({ dicPath: KUROMOJI_DICT }).build((reason, built) => {
          if (reason || !built) reject(reason ?? new Error("辞書を初期化できませんでした"));
          else resolve(built);
        });
      }), 90_000, "日本語辞書の読み込みが90秒以内に完了しませんでした");
    })().catch((reason) => {
      kuromojiTokenizerPromise = null;
      throw reason;
    });
  }
  const tokenizer = await kuromojiTokenizerPromise;
  endpoint.postMessage({ type: "progress", stage: "align", percent: 56, message: "日本語辞書を読み込みました" });
  return (text: string) => normalizeJapaneseTokens(tokenizer.tokenize(text));
}

async function createAnalysisRuntime(): Promise<AnalysisRuntime> {
  endpoint.postMessage({ type: "progress", stage: "align", percent: 56, message: "音素解析エンジンを準備しています" });
  const ort = await withTimeout(
    import("onnxruntime-web/wasm"),
    60_000,
    "音声解析エンジンの準備が60秒以内に完了しませんでした。ページを再読み込みして再実行してください",
  );
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
  const modelBytes = await fetchAlignmentModel();
  endpoint.postMessage({ type: "progress", stage: "align", percent: 68, message: "音素モデルのセッションを準備しています" });
  const session = await withTimeout(
    ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] }),
    300_000,
    "音素モデルの準備が5分以内に完了しませんでした。ブラウザのメモリ不足の可能性があります",
  );
  return { ort, session };
}

async function getAnalysisRuntime() {
  if (!runtimePromise) {
    runtimePromise = createAnalysisRuntime().catch((reason) => {
      runtimePromise = null;
      throw reason;
    });
  } else {
    endpoint.postMessage({
      type: "progress",
      stage: "align",
      percent: 68,
      message: "メモリに保持した日本語 HuBERT 音素モデルを再利用しています",
    });
  }
  return runtimePromise;
}

endpoint.onmessage = async ({ data }) => {
  if (data.type !== "analyze") return;

  try {
    const tokenizeForAlignment = await getTokenizeForAlignment(data.text);
    const { ort, session } = await getAnalysisRuntime();
    const phonemizer = new PhonemizerJa({ tokenize: tokenizeForAlignment }, PHONE_VOCAB_JA);
    const result = await alignWholeRecording(session, ort, phonemizer, data.audio, data.text);
    const tokens = tokenizeForAlignment(data.text);
    const targetToWord = new Int32Array(result.phoneIds.length).fill(-1);
    const wordTargetStarts: number[] = [];
    let targetOffset = 0;
    result.groupLens.forEach((length, wordIndex) => {
      wordTargetStarts.push(targetOffset);
      for (let index = targetOffset; index < targetOffset + length; index += 1) targetToWord[index] = wordIndex;
      targetOffset += length;
    });
    const firstFrames = new Int32Array(result.sourceWords.length).fill(-1);
    const lastFrames = new Int32Array(result.sourceWords.length).fill(-1);
    result.frameTokens.forEach((targetIndex, frameIndex) => {
      if (targetIndex < 0) return;
      const wordIndex = targetToWord[targetIndex];
      if (wordIndex < 0) return;
      if (firstFrames[wordIndex] < 0) firstFrames[wordIndex] = frameIndex;
      lastFrames[wordIndex] = frameIndex;
    });

    const words: Segment[] = [];
    const wordReadings: WordReading[] = [];
    let omittedMoras = 0;
    result.sourceWords.forEach((surface, wordIndex) => {
      const exact = tokens[wordIndex];
      const token: PronounceableJapaneseToken | undefined = exact?.surface === surface
        ? exact
        : tokens.find((candidate, index) => index >= wordIndex && candidate.surface === surface);
      const reading = token?.reading ?? fallbackReadingForAlignedSurface(surface);
      const moras = splitJapaneseMoras(reading);
      const targetStart = wordTargetStarts[wordIndex] ?? 0;
      const targetEnd = targetStart + (result.groupLens[wordIndex] ?? 0);
      const nuclei: number[] = [];
      for (let targetIndex = targetStart; targetIndex < targetEnd; targetIndex += 1) {
        const label = phonemizer.id2phone[String(result.phoneIds[targetIndex])] ?? "";
        if (MORA_PHONES.has(label)) nuclei.push(targetIndex);
      }
      const moraSupport = moras.map((_, moraIndex) => {
        if (!nuclei.length) {
          const supported = Array.from(result.supportedTargets.slice(targetStart, targetEnd)).some(Boolean);
          return { acoustic: supported, pitch: false };
        }
        const nucleusIndex = moras.length === nuclei.length
          ? moraIndex
          : Math.min(nuclei.length - 1, Math.floor(((moraIndex + 0.5) * nuclei.length) / Math.max(1, moras.length)));
        const nucleusTarget = nuclei[nucleusIndex];
        const moraTargetStart = nucleusIndex === 0 ? targetStart : nuclei[nucleusIndex - 1] + 1;
        const acoustic = Array.from(result.supportedTargets.slice(moraTargetStart, nucleusTarget + 1)).some(Boolean);
        return { acoustic, pitch: Boolean(result.supportedTargets[nucleusTarget]) };
      });
      const keptMoras: string[] = [];
      const moraPitchSupported: boolean[] = [];
      moras.forEach((mora, moraIndex) => {
        const support = moraSupport[moraIndex];
        if (!support?.acoustic) return;
        keptMoras.push(mora);
        moraPitchSupported.push(support.pitch);
      });
      omittedMoras += moras.length - keptMoras.length;
      if (!keptMoras.length || firstFrames[wordIndex] < 0) return;
      words.push([
        firstFrames[wordIndex] * result.resolution,
        (lastFrames[wordIndex] + 1) * result.resolution,
        surface,
      ]);
      wordReadings.push({ surface, reading, moras: keptMoras, moraPitchSupported });
    });
    const aligned = { phones: result.phones, words, phoneIds: result.phoneIds, wordReadings, omittedMoras };
    endpoint.postMessage({
      type: "progress",
      stage: "align",
      percent: 84,
      message: omittedMoras
        ? `音響上見つからない ${omittedMoras} モーラを省略しました`
        : "モーラごとの読みを割り当てています",
    });

    endpoint.postMessage({ type: "progress", stage: "pitch", percent: 87, message: "SwiftF0で声の音高を検出しています" });
    let swiftFrames: PitchFrame[] = [];
    try {
      swiftFrames = await analyzeSwiftF0(ort, data.audio);
    } catch (reason) {
      console.warn("SwiftF0 failed; continuing with YIN fallback", reason);
      endpoint.postMessage({ type: "progress", stage: "pitch", percent: 91, message: "SwiftF0を利用できないためYINで解析を継続しています" });
    }
    endpoint.postMessage({ type: "progress", stage: "pitch", percent: 92, message: "YINで不確実な音高を再確認しています" });
    const yinFrames = analyzePitch(data.audio);
    const pitchFrames = mergePitchFrames(swiftFrames, yinFrames);
    endpoint.postMessage({ type: "result", percent: 96, aligned, pitchFrames });
  } catch (reason) {
    endpoint.postMessage({
      type: "error",
      message: reason instanceof Error ? reason.message : String(reason),
    });
  }
};
