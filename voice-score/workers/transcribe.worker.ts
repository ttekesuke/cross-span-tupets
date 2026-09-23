import type { Tensor } from "onnxruntime-web";

type TranscribeRequest = { type: "transcribe"; audio: Float32Array };
type TranscriptChunk = { text: string; timestamp: [number | null, number | null] };
type TransformersModule = {
  pipeline: (
    task: string,
    model: string,
    options: Record<string, unknown>,
  ) => Promise<(audio: Float32Array, options: Record<string, unknown>) => Promise<{
    text: string;
    chunks?: TranscriptChunk[];
  }>>;
  env?: { allowLocalModels?: boolean };
};
type WorkerEndpoint = {
  onmessage: ((event: MessageEvent<TranscribeRequest>) => void) | null;
  postMessage: (message: unknown) => void;
};
type SpeechRegion = {
  start: number;
  end: number;
  meanProbability: number;
  peakProbability: number;
  activeRatio: number;
};

const endpoint = self as unknown as WorkerEndpoint;
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const WHISPER_MODEL = "onnx-community/whisper-base";
const VAD_MODEL = "https://huggingface.co/openEuler/silero-vad/resolve/main/assets/silero_vad.onnx?download=true";
const SAMPLE_RATE = 16_000;
const VAD_FRAME_SAMPLES = 512;
const VAD_CONTEXT_SAMPLES = 64;
const VAD_START_THRESHOLD = 0.30;
const VAD_END_THRESHOLD = 0.20;
const VAD_MIN_ACTIVE_FRAMES = 4;
const VAD_PRE_ROLL_FRAMES = 7;
const VAD_END_HANG_FRAMES = 13;
const VAD_MERGE_GAP_SECONDS = 0.8;

function postProgress(percent: number, message: string) {
  endpoint.postMessage({ type: "progress", percent, message });
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function frameRms(frame: Float32Array) {
  let sum = 0;
  for (let index = 0; index < frame.length; index += 1) sum += frame[index] * frame[index];
  return Math.sqrt(sum / Math.max(1, frame.length));
}

function mergeSpeechRegions(regions: SpeechRegion[]) {
  const merged: SpeechRegion[] = [];
  for (const region of regions) {
    const previous = merged.at(-1);
    if (!previous || region.start - previous.end > VAD_MERGE_GAP_SECONDS) {
      merged.push({ ...region });
      continue;
    }
    const previousDuration = previous.end - previous.start;
    const regionDuration = region.end - region.start;
    const duration = previousDuration + regionDuration;
    previous.end = Math.max(previous.end, region.end);
    previous.meanProbability = duration > 0
      ? ((previous.meanProbability * previousDuration) + (region.meanProbability * regionDuration)) / duration
      : Math.max(previous.meanProbability, region.meanProbability);
    previous.peakProbability = Math.max(previous.peakProbability, region.peakProbability);
    previous.activeRatio = duration > 0
      ? ((previous.activeRatio * previousDuration) + (region.activeRatio * regionDuration)) / duration
      : Math.max(previous.activeRatio, region.activeRatio);
  }
  return merged;
}

async function detectSpeech(audio: Float32Array): Promise<SpeechRegion[]> {
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
  const response = await fetch(VAD_MODEL);
  if (!response.ok) throw new Error(`発話検出モデルを読み込めませんでした（HTTP ${response.status}）`);
  const session = await ort.InferenceSession.create(await response.arrayBuffer(), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const sampleRateTensor = new ort.Tensor("int64", [BigInt(SAMPLE_RATE)]);
  let state: Tensor = new ort.Tensor("float32", new Float32Array(2 * 128), [2, 1, 128]);
  let context = new Float32Array(VAD_CONTEXT_SAMPLES);
  const probabilities: number[] = [];
  const energies: number[] = [];
  const frameCount = Math.ceil(audio.length / VAD_FRAME_SAMPLES);

  try {
    for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
      const frame = new Float32Array(VAD_FRAME_SAMPLES);
      const source = audio.subarray(
        frameIndex * VAD_FRAME_SAMPLES,
        Math.min(audio.length, (frameIndex + 1) * VAD_FRAME_SAMPLES),
      );
      frame.set(source);
      const withContext = new Float32Array(VAD_CONTEXT_SAMPLES + VAD_FRAME_SAMPLES);
      withContext.set(context);
      withContext.set(frame, VAD_CONTEXT_SAMPLES);
      context = frame.slice(-VAD_CONTEXT_SAMPLES);
      const input = new ort.Tensor("float32", withContext, [1, withContext.length]);
      const previousState = state;
      const output = await session.run({ input, state, sr: sampleRateTensor });
      const probability = Number(output.output?.data[0] ?? 0);
      const nextState = output.stateN;
      if (!nextState) throw new Error("発話検出モデルから状態を取得できませんでした");
      state = nextState;
      input.dispose();
      previousState.dispose();
      output.output?.dispose();
      probabilities.push(clamp(probability, 0, 1));
      energies.push(frameRms(frame));
      if (frameIndex % 50 === 0 || frameIndex === frameCount - 1) {
        const progress = (frameIndex + 1) / Math.max(1, frameCount);
        postProgress(
          12 + Math.round(progress * 8),
          `人の声がある区間を検出しています（${Math.round(progress * 100)}%）`,
        );
      }
    }
  } finally {
    state.dispose();
    sampleRateTensor.dispose();
    await session.release();
  }

  const sortedEnergy = [...energies].sort((a, b) => a - b);
  const noiseFloor = sortedEnergy[Math.floor(sortedEnergy.length * 0.2)] ?? 0;
  const energyGate = Math.max(0.0008, noiseFloor * 1.8);
  const regions: SpeechRegion[] = [];
  let startFrame = -1;
  let activeFrames = 0;
  let belowFrames = 0;
  let probabilitySum = 0;
  let peakProbability = 0;
  let observedFrames = 0;

  const finishRegion = (endFrame: number) => {
    if (startFrame < 0) return;
    if (activeFrames >= VAD_MIN_ACTIVE_FRAMES) {
      const paddedStart = Math.max(0, startFrame - VAD_PRE_ROLL_FRAMES);
      const paddedEnd = Math.min(frameCount, endFrame + 2);
      regions.push({
        start: (paddedStart * VAD_FRAME_SAMPLES) / SAMPLE_RATE,
        end: Math.min(audio.length / SAMPLE_RATE, (paddedEnd * VAD_FRAME_SAMPLES) / SAMPLE_RATE),
        meanProbability: probabilitySum / Math.max(1, observedFrames),
        peakProbability,
        activeRatio: activeFrames / Math.max(1, observedFrames),
      });
    }
    startFrame = -1;
    activeFrames = 0;
    belowFrames = 0;
    probabilitySum = 0;
    peakProbability = 0;
    observedFrames = 0;
  };

  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = probabilities[index];
    const hasEnergy = energies[index] >= energyGate;
    const startsSpeech = probability >= VAD_START_THRESHOLD && hasEnergy;
    if (startFrame < 0) {
      if (!startsSpeech) continue;
      startFrame = index;
    }
    observedFrames += 1;
    probabilitySum += probability;
    peakProbability = Math.max(peakProbability, probability);
    if (startsSpeech) activeFrames += 1;
    if (probability < VAD_END_THRESHOLD || !hasEnergy) belowFrames += 1;
    else belowFrames = 0;
    if (belowFrames >= VAD_END_HANG_FRAMES) finishRegion(index - belowFrames + 1);
  }
  finishRegion(probabilities.length);
  return mergeSpeechRegions(regions);
}

const HALLUCINATION_PHRASES = [
  "ご視聴ありがとうございました",
  "ご視聴いただきありがとうございました",
  "ご視聴頂きありがとうございました",
  "チャンネル登録をお願いします",
  "チャンネル登録よろしくお願いします",
  "最後までご視聴ありがとうございました",
];

function normalizedText(text: string) {
  return text.normalize("NFKC").replace(/[\s。、，,.!！?？「」『』（）()\-ー〜~♪♫…]/gu, "");
}

function hasDegenerateRepetition(text: string) {
  const normalized = normalizedText(text);
  if (normalized.length < 6) return false;
  for (let unitLength = 1; unitLength <= Math.min(12, Math.floor(normalized.length / 3)); unitLength += 1) {
    for (let start = 0; start + (unitLength * 3) <= normalized.length; start += 1) {
      const unit = normalized.slice(start, start + unitLength);
      if (normalized.slice(start + unitLength, start + (unitLength * 2)) === unit
        && normalized.slice(start + (unitLength * 2), start + (unitLength * 3)) === unit) return true;
    }
  }
  return false;
}

function shouldSuppress(text: string, region: SpeechRegion) {
  const normalized = normalizedText(text);
  if (!normalized || /^[♪♫]+$/u.test(text.trim())) return true;
  const weakSpeechEvidence = region.meanProbability < 0.34
    || region.activeRatio < 0.22
    || region.peakProbability < 0.55;
  const knownAttractor = HALLUCINATION_PHRASES.some((phrase) => normalized === normalizedText(phrase));
  const duration = Math.max(0.1, region.end - region.start);
  const implausiblyDense = normalized.length / duration > 18;
  return (knownAttractor && weakSpeechEvidence)
    || (hasDegenerateRepetition(text) && weakSpeechEvidence)
    || implausiblyDense;
}

endpoint.onmessage = async ({ data }) => {
  if (data.type !== "transcribe") return;
  try {
    postProgress(10, "Silero VAD を準備しています");
    const speechRegions = await detectSpeech(data.audio);
    if (!speechRegions.length) {
      throw new Error("人の声として判定できる区間がありませんでした。解析範囲を声が入っている部分に絞るか、音量を確認してください");
    }
    postProgress(21, `発話区間を ${speechRegions.length} 個検出しました`);

    const transformers = await import(/* @vite-ignore */ TRANSFORMERS_URL) as TransformersModule;
    if (transformers.env) transformers.env.allowLocalModels = false;
    const recognizer = await transformers.pipeline("automatic-speech-recognition", WHISPER_MODEL, {
      dtype: "q8",
      device: "wasm",
      session_options: { graphOptimizationLevel: "basic" },
      progress_callback: (progress: { status?: string; progress?: number }) => {
        if (progress.status !== "progress" || typeof progress.progress !== "number") return;
        postProgress(
          22 + Math.round(progress.progress * 0.14),
          `Whisper モデルを取得しています（${Math.round(progress.progress)}%）`,
        );
      },
    });

    const acceptedTexts: string[] = [];
    const acceptedChunks: Array<{ text: string; start: number; end: number }> = [];
    let suppressedCount = 0;
    const duration = data.audio.length / SAMPLE_RATE;
    for (let index = 0; index < speechRegions.length; index += 1) {
      const region = speechRegions[index];
      postProgress(
        37 + Math.round((index / speechRegions.length) * 7),
        `発話区間を文字起こししています（${index + 1}/${speechRegions.length}）`,
      );
      const startSample = Math.floor(region.start * SAMPLE_RATE);
      const endSample = Math.min(data.audio.length, Math.ceil(region.end * SAMPLE_RATE));
      const result = await recognizer(data.audio.slice(startSample, endSample), {
        language: "japanese",
        task: "transcribe",
        return_timestamps: true,
        chunk_length_s: 24,
        stride_length_s: 3,
        force_full_sequences: false,
      });
      const text = result.text.trim();
      if (shouldSuppress(text, region)) {
        suppressedCount += 1;
        continue;
      }
      acceptedTexts.push(text);
      const chunks = (result.chunks ?? []).flatMap((chunk) => {
        const chunkText = chunk.text.trim();
        const rawStart = chunk.timestamp?.[0];
        const rawEnd = chunk.timestamp?.[1];
        const localStart = typeof rawStart === "number" && Number.isFinite(rawStart) ? rawStart : 0;
        const localEnd = typeof rawEnd === "number" && Number.isFinite(rawEnd)
          ? rawEnd
          : region.end - region.start;
        const start = clamp(region.start + localStart, region.start, duration);
        const end = clamp(region.start + localEnd, start, duration);
        return chunkText && end > start ? [{ text: chunkText, start, end }] : [];
      });
      if (chunks.length) acceptedChunks.push(...chunks);
      else if (text) acceptedChunks.push({ text, start: region.start, end: region.end });
    }

    if (!acceptedTexts.length) {
      throw new Error("音声らしい区間はありましたが、幻聴の可能性が高い文字起こしだけだったため除外しました。声が明瞭な範囲に絞って再実行してください");
    }
    postProgress(44, suppressedCount
      ? `文字起こしを確認し、疑わしい ${suppressedCount} 区間を除外しました`
      : "文字起こしの幻聴チェックが完了しました");
    endpoint.postMessage({
      type: "result",
      percent: 45,
      text: acceptedTexts.join(""),
      chunks: acceptedChunks,
    });
  } catch (reason) {
    endpoint.postMessage({
      type: "error",
      message: reason instanceof Error ? reason.message : String(reason),
    });
  }
};
