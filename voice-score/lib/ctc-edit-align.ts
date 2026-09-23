const NEGATIVE_INFINITY = -1e30;
const MAX_TRACEBACK_CELLS = 50_000_000;
const AUDIO_INSERTION_PENALTY = 5;
const NUCLEUS_DELETION_PENALTY = 6.5;
const CONSONANT_DELETION_PENALTY = 4.5;
const CLOSURE_DELETION_PENALTY = 3.5;
const NUCLEI = new Set(["a", "i", "u", "e", "o", "N"]);

const A_EXTEND = 1;
const A_BLANK = 2;
const A_MATCH_FROM_A = 3;
const A_MATCH_FROM_D = 4;
const D_BLANK = 1;
const D_DELETE_FROM_A = 2;
const D_DELETE_FROM_D = 3;

export type CtcEditAlignment = {
  frameTokens: Int32Array;
  matchedTargets: Uint8Array;
  supportedTargets: Uint8Array;
};

function deletionPenalty(label: string) {
  if (label === "cl") return CLOSURE_DELETION_PENALTY;
  return NUCLEI.has(label) ? NUCLEUS_DELETION_PENALTY : CONSONANT_DELETION_PENALTY;
}

/**
 * Memory-efficient monotonic CTC/edit alignment.
 *
 * Unlike strict CTC forced alignment, this path may delete transcript phones
 * and may consume untranscribed audio. Only one byte of traceback state is
 * retained per time/target cell; score rows use O(targets) memory.
 */
export function ctcEditAlign(
  logits: ArrayLike<number>,
  frames: number,
  vocab: number,
  targetIds: number[],
  blank: number,
  id2phone: Record<string, string>,
): CtcEditAlignment {
  const targetCount = targetIds.length;
  const width = targetCount + 1;
  const cells = (frames + 1) * width;
  if (cells > MAX_TRACEBACK_CELLS) {
    throw new Error("音声と歌詞が非常に長く、ブラウザの安全上限を超えました。10分程度を目安に範囲を分けてください");
  }

  const pointers = new Uint8Array(cells);
  let previousA = new Float64Array(width).fill(NEGATIVE_INFINITY);
  let previousD = new Float64Array(width).fill(NEGATIVE_INFINITY);
  let currentA = new Float64Array(width).fill(NEGATIVE_INFINITY);
  let currentD = new Float64Array(width).fill(NEGATIVE_INFINITY);
  previousD[0] = 0;
  for (let target = 1; target <= targetCount; target += 1) {
    const label = id2phone[String(targetIds[target - 1])] ?? "";
    previousD[target] = previousD[target - 1] - deletionPenalty(label);
    pointers[target] = D_DELETE_FROM_D << 3;
  }

  for (let frame = 1; frame <= frames; frame += 1) {
    const rowOffset = (frame - 1) * vocab;
    let bestNonBlank = NEGATIVE_INFINITY;
    for (let phone = 0; phone < vocab; phone += 1) {
      if (phone !== blank && logits[rowOffset + phone] > bestNonBlank) bestNonBlank = logits[rowOffset + phone];
    }
    const blankScore = Math.max(logits[rowOffset + blank], bestNonBlank - AUDIO_INSERTION_PENALTY);
    const pointerOffset = frame * width;

    currentA[0] = NEGATIVE_INFINITY;
    currentD[0] = previousD[0] + blankScore;
    pointers[pointerOffset] = D_BLANK << 3;

    for (let target = 1; target <= targetCount; target += 1) {
      const phoneId = targetIds[target - 1];
      const phoneScore = logits[rowOffset + phoneId];

      let aScore = previousD[target - 1] + phoneScore;
      let aPointer = A_MATCH_FROM_D;
      const matchFromA = previousA[target - 1] + phoneScore;
      if (matchFromA > aScore) { aScore = matchFromA; aPointer = A_MATCH_FROM_A; }
      const extend = previousA[target] + phoneScore;
      if (extend > aScore) { aScore = extend; aPointer = A_EXTEND; }
      const blankFromA = previousA[target] + blankScore;
      if (blankFromA > aScore) { aScore = blankFromA; aPointer = A_BLANK; }
      currentA[target] = aScore;

      const label = id2phone[String(phoneId)] ?? "";
      const penalty = deletionPenalty(label);
      let dScore = previousD[target] + blankScore;
      let dPointer = D_BLANK;
      const deleteFromA = currentA[target - 1] - penalty;
      if (deleteFromA > dScore) { dScore = deleteFromA; dPointer = D_DELETE_FROM_A; }
      const deleteFromD = currentD[target - 1] - penalty;
      if (deleteFromD > dScore) { dScore = deleteFromD; dPointer = D_DELETE_FROM_D; }
      currentD[target] = dScore;
      pointers[pointerOffset + target] = aPointer | (dPointer << 3);
    }

    [previousA, currentA] = [currentA, previousA];
    [previousD, currentD] = [currentD, previousD];
  }

  const frameTokens = new Int32Array(frames).fill(-1);
  let frame = frames;
  let target = targetCount;
  let state: "A" | "D" = previousA[targetCount] >= previousD[targetCount] ? "A" : "D";
  while (frame > 0 || target > 0) {
    const pointer = pointers[frame * width + target];
    if (state === "A") {
      const operation = pointer & 7;
      if (operation === A_EXTEND) {
        frameTokens[frame - 1] = target - 1;
        frame -= 1;
      } else if (operation === A_BLANK) {
        frame -= 1;
      } else if (operation === A_MATCH_FROM_A || operation === A_MATCH_FROM_D) {
        frameTokens[frame - 1] = target - 1;
        frame -= 1;
        target -= 1;
        state = operation === A_MATCH_FROM_A ? "A" : "D";
      } else {
        throw new Error("音素整列の経路を復元できませんでした");
      }
    } else {
      const operation = pointer >> 3;
      if (operation === D_BLANK) {
        frame -= 1;
      } else if (operation === D_DELETE_FROM_A || operation === D_DELETE_FROM_D) {
        target -= 1;
        state = operation === D_DELETE_FROM_A ? "A" : "D";
      } else {
        throw new Error("音素整列の欠落経路を復元できませんでした");
      }
    }
  }

  const matchedTargets = new Uint8Array(targetCount);
  const bestMargins = new Float32Array(targetCount).fill(NEGATIVE_INFINITY);
  for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
    const targetIndex = frameTokens[frameIndex];
    if (targetIndex < 0) continue;
    matchedTargets[targetIndex] = 1;
    const rowOffset = frameIndex * vocab;
    const phoneId = targetIds[targetIndex];
    let competitor = NEGATIVE_INFINITY;
    for (let phone = 0; phone < vocab; phone += 1) {
      if (phone !== phoneId && logits[rowOffset + phone] > competitor) competitor = logits[rowOffset + phone];
    }
    bestMargins[targetIndex] = Math.max(bestMargins[targetIndex], logits[rowOffset + phoneId] - competitor);
  }

  const supportedTargets = matchedTargets.slice();
  for (let targetIndex = 0; targetIndex < targetCount; targetIndex += 1) {
    if (!matchedTargets[targetIndex]) continue;
    const label = id2phone[String(targetIds[targetIndex])] ?? "";
    const minimumMargin = NUCLEI.has(label) ? -1 : -2;
    if (bestMargins[targetIndex] < minimumMargin) supportedTargets[targetIndex] = 0;
  }
  return { frameTokens, matchedTargets, supportedTargets };
}
