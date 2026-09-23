import { YIN } from "pitchfinder";

export type PitchSource = "swiftf0" | "yin" | "interpolated" | "carried";
export type NotePitchSource = PitchSource | "unvoiced";

export type PitchFrame = {
  time: number;
  frequency: number | null;
  midi: number | null;
  rms: number;
  confidence?: number;
  source?: PitchSource;
};

export type Segment = [number, number, string];

export type WordReading = {
  surface: string;
  reading: string;
  moras: string[];
  /** False when the mora is present acoustically but its pitch-bearing nucleus is not. */
  moraPitchSupported?: boolean[];
};

export type ScoreNote = {
  start: number;
  end: number;
  midi: number | null;
  lyric: string;
  phoneme: string;
  frequency: number | null;
  pitchSource: NotePitchSource;
  confidence: number | null;
  notehead: "normal" | "x";
};

const NUCLEI = new Set(["a", "i", "u", "e", "o", "N"]);
const SCORE_PHONES = new Set([...NUCLEI, "cl"]);
const SMALL_KANA = new Set(["ぁ", "ぃ", "ぅ", "ぇ", "ぉ", "ゃ", "ゅ", "ょ", "ゎ", "ゕ", "ゖ"]);

function katakanaToHiragana(value: string) {
  return value.normalize("NFKC").replace(/[ァ-ヶ]/g, (character) =>
    String.fromCharCode(character.charCodeAt(0) - 0x60));
}

/** Split a Japanese reading into sung morae: きょう -> [きょ, う]. */
export function splitJapaneseMoras(reading: string) {
  const moras: string[] = [];
  for (const character of katakanaToHiragana(reading)) {
    if (SMALL_KANA.has(character) && moras.length) {
      moras[moras.length - 1] += character;
    } else if (/^[ぁ-ゖー]$/u.test(character)) {
      moras.push(character);
    }
  }
  return moras;
}

export function hzToMidi(hz: number) {
  return 69 + 12 * Math.log2(hz / 440);
}

export function midiToName(midi: number) {
  const names = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"];
  const rounded = Math.round(midi * 2) / 2;
  const semitone = Math.floor(rounded);
  const quarterTone = rounded - semitone >= 0.25 ? "+50¢" : "";
  return `${names[((semitone % 12) + 12) % 12]}${quarterTone}${Math.floor(semitone / 12) - 1}`;
}

function quantizeQuarterTone(midi: number) {
  return Math.round(midi * 2) / 2;
}

function percentile(values: number[], ratio: number) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * ratio))];
}

/** Remove rumble/DC, soften hiss and bring quiet recordings into a useful range. */
export function preparePitchAudio(waveform: Float32Array, sampleRate = 16000) {
  if (!waveform.length) return waveform;
  let mean = 0;
  for (let index = 0; index < waveform.length; index += 1) mean += waveform[index];
  mean /= waveform.length;

  const highPass = new Float32Array(waveform.length);
  const highPassAlpha = 1 / (1 + (2 * Math.PI * 65) / sampleRate);
  let previousInput = waveform[0] - mean;
  let previousHigh = 0;
  let low = 0;
  const lowPassAlpha = (2 * Math.PI * 1800) / (sampleRate + 2 * Math.PI * 1800);
  let sum = 0;
  for (let index = 0; index < waveform.length; index += 1) {
    const input = waveform[index] - mean;
    const high = highPassAlpha * (previousHigh + input - previousInput);
    low += lowPassAlpha * (high - low);
    highPass[index] = low;
    sum += low * low;
    previousInput = input;
    previousHigh = high;
  }

  const rms = Math.sqrt(sum / waveform.length);
  const gain = Math.min(18, Math.max(0.75, 0.14 / Math.max(rms, 0.0001)));
  const enhanced = new Float32Array(waveform.length);
  for (let index = 0; index < highPass.length; index += 1) {
    // Soft limiting avoids turning gain-normalized peaks into square waves.
    enhanced[index] = Math.tanh(highPass[index] * gain * 1.25) / Math.tanh(1.25);
  }
  return enhanced;
}

export function analyzePitch(waveform: Float32Array, sampleRate = 16000): PitchFrame[] {
  const windowSize = 2048;
  const hopSize = 320;
  const detectorSampleRate = sampleRate / 2;
  const detector = YIN({ sampleRate: detectorSampleRate, threshold: 0.16, probabilityThreshold: 0.62 });
  const enhanced = preparePitchAudio(waveform, sampleRate);
  const candidates: Array<{ time: number; slice: Float32Array; rms: number }> = [];

  for (let offset = 0; offset + windowSize <= enhanced.length; offset += hopSize) {
    const source = enhanced.subarray(offset, offset + windowSize);
    const slice = new Float32Array(windowSize / 2);
    let sum = 0;
    for (let index = 0; index < source.length; index += 1) sum += source[index] * source[index];
    for (let index = 0; index < slice.length; index += 1) slice[index] = source[index * 2];
    candidates.push({
      time: (offset + windowSize / 2) / sampleRate,
      slice,
      rms: Math.sqrt(sum / source.length),
    });
  }

  const rmsValues = candidates.map((candidate) => candidate.rms);
  const noiseFloor = percentile(rmsValues, 0.18);
  const typicalLevel = percentile(rmsValues, 0.5);
  // A cropped range may contain singing throughout, so never assume that its
  // quietest frames are pure noise. Cap the gate relative to the median level.
  const voicedThreshold = Math.max(0.0025, Math.min(noiseFloor * 1.55, typicalLevel * 0.35));
  const frames = candidates.map(({ time, slice, rms }) => {
    const found = rms >= voicedThreshold ? detector(slice) : null;
    const frequency = found && found >= 55 && found <= 1400 ? found : null;
    return {
      time,
      frequency,
      midi: frequency ? hzToMidi(frequency) : null,
      rms,
      source: frequency ? "yin" as const : undefined,
    };
  });

  return removePitchOutliers(frames);
}

/** Use SwiftF0 first, then recover only its uncertain frames with YIN. */
export function mergePitchFrames(swiftFrames: PitchFrame[], yinFrames: PitchFrame[]) {
  if (!swiftFrames.length) return yinFrames;
  const swiftRms = swiftFrames.map((frame) => frame.rms);
  const reliableRms = swiftFrames
    .filter((frame) => frame.midi !== null && (frame.confidence ?? 0) >= 0.5)
    .map((frame) => frame.rms);
  const noiseFloor = percentile(swiftRms, 0.18);
  const typicalVoiceLevel = percentile(reliableRms.length ? reliableRms : swiftRms, 0.5);
  const activeThreshold = Math.max(0.002, Math.min(noiseFloor * 1.8, typicalVoiceLevel * 0.28));
  let yinIndex = 0;
  let lastVoicedMidi: number | null = null;
  let lastVoicedTime = -Infinity;
  const merged = swiftFrames.map((swift) => {
    while (
      yinIndex + 1 < yinFrames.length
      && Math.abs(yinFrames[yinIndex + 1].time - swift.time) < Math.abs(yinFrames[yinIndex].time - swift.time)
    ) yinIndex += 1;
    const yin = yinFrames[yinIndex];
    const confidence = swift.confidence ?? 0;
    const swiftMidi = swift.midi;
    const recentlyVoiced = swift.time - lastVoicedTime <= 0.2;
    const swiftThreshold = recentlyVoiced ? 0.24 : 0.44;
    const swiftIsVoiced = swiftMidi !== null && confidence >= swiftThreshold;
    const yinIsNear = yin?.midi !== null && Math.abs((yin?.time ?? Infinity) - swift.time) <= 0.035;
    const detectorsAgree = swiftMidi !== null && yinIsNear
      && Math.abs(swiftMidi - (yin.midi as number)) <= 2.5;
    const continuousLowConfidenceVoice = swiftMidi !== null
      && confidence >= 0.14
      && swift.rms >= activeThreshold
      && lastVoicedMidi !== null
      && swift.time - lastVoicedTime <= 0.34
      && Math.abs(swiftMidi - lastVoicedMidi) <= 3.5;

    if (swiftIsVoiced || (confidence >= 0.2 && detectorsAgree) || continuousLowConfidenceVoice) {
      lastVoicedMidi = swiftMidi;
      lastVoicedTime = swift.time;
      return { ...swift, source: "swiftf0" as const };
    }
    if (yinIsNear) {
      lastVoicedMidi = yin.midi;
      lastVoicedTime = swift.time;
      return {
        ...swift,
        frequency: yin.frequency,
        midi: yin.midi,
        confidence: Math.max(confidence, 0.35),
        source: "yin" as const,
      };
    }
    return { ...swift, frequency: null, midi: null, source: undefined };
  });
  return removePitchOutliers(merged);
}

function removePitchOutliers(frames: PitchFrame[]) {
  const corrected = frames.map((frame, index) => {
    if (frame.midi === null) return frame;
    const neighbours = frames
      .slice(Math.max(0, index - 3), index + 4)
      .map((item) => item.midi)
      .filter((value): value is number => value !== null)
      .sort((a, b) => a - b);
    if (neighbours.length < 3) return frame;
    const median = neighbours[Math.floor(neighbours.length / 2)];
    const octaveShift = Math.round((median - frame.midi) / 12) * 12;
    const octaveCorrected = frame.midi + octaveShift;
    if (Math.abs(octaveCorrected - median) <= 2.5 && octaveShift !== 0) {
      return {
        ...frame,
        midi: octaveCorrected,
        frequency: 440 * 2 ** ((octaveCorrected - 69) / 12),
      };
    }
    if (Math.abs(frame.midi - median) > 8) {
      return { ...frame, frequency: null, midi: null };
    }
    return frame;
  });

  // Bridge only very short drop-outs between matching pitches (at most 80 ms).
  for (let start = 0; start < corrected.length; start += 1) {
    if (corrected[start].midi !== null) continue;
    let end = start;
    while (end < corrected.length && corrected[end].midi === null) end += 1;
    const before = corrected[start - 1]?.midi;
    const after = corrected[end]?.midi;
    if (end - start <= 4 && before !== null && before !== undefined && after !== null && after !== undefined && Math.abs(before - after) <= 2.5) {
      for (let index = start; index < end; index += 1) {
        const progress = (index - start + 1) / (end - start + 1);
        const midi = before + (after - before) * progress;
        corrected[index] = {
          ...corrected[index],
          midi,
          frequency: 440 * 2 ** ((midi - 69) / 12),
          source: "interpolated",
        };
      }
    }
    start = end - 1;
  }
  return corrected;
}

function median(values: number[]) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function notesFromAlignment(
  phones: Segment[],
  words: Segment[],
  pitchFrames: PitchFrame[],
  wordReadings: WordReading[] = [],
): ScoreNote[] {
  const notes: ScoreNote[] = [];
  const rmsValues = pitchFrames.map((frame) => frame.rms);
  const voicedRms = pitchFrames.filter((frame) => frame.midi !== null).map((frame) => frame.rms);
  const noiseFloor = percentile(rmsValues, 0.18);
  const typicalVoiceLevel = percentile(voicedRms.length ? voicedRms : rmsValues, 0.5);
  const activityThreshold = Math.max(0.002, Math.min(noiseFloor * 1.8, typicalVoiceLevel * 0.28));
  const frameStep = median(pitchFrames.slice(1).map((frame, index) => frame.time - pitchFrames[index].time)) ?? 0.02;

  const frameIsActive = (index: number) => {
    const frame = pitchFrames[index];
    if (!frame) return false;
    if (frame.midi !== null) return true;
    let energeticNeighbours = 0;
    for (let offset = -1; offset <= 1; offset += 1) {
      if ((pitchFrames[index + offset]?.rms ?? 0) >= activityThreshold) energeticNeighbours += 1;
    }
    return energeticNeighbours >= 2;
  };

  const activityInRange = (start: number, end: number) => {
    const indices = pitchFrames.flatMap((frame, index) =>
      frame.time >= start && frame.time <= end ? [index] : []);
    const activeCount = indices.filter(frameIsActive).length;
    return {
      indices,
      activeCount,
      isActive: activeCount > 0 && (indices.length <= 3 || activeCount >= 2 || activeCount / indices.length >= 0.25),
    };
  };

  const trimTrailingSilence = (start: number, end: number) => {
    const { indices } = activityInRange(start, end);
    const lastActiveIndex = indices.findLast(frameIsActive);
    if (lastActiveIndex === undefined) return end;
    const activityEnd = Math.min(end, pitchFrames[lastActiveIndex].time + frameStep * 1.5);
    return end - activityEnd >= 0.08
      ? Math.max(start + 0.045, activityEnd)
      : end;
  };

  const addNote = (start: number, end: number, lyric: string, phoneme: string, forceUnknownPitch = false) => {
    const acousticEnd = trimTrailingSilence(start, end);
    const paddedStart = Math.max(0, start - 0.035);
    const paddedEnd = acousticEnd + 0.035;
    const inMora = pitchFrames.filter(
      (frame) => frame.time >= paddedStart && frame.time <= paddedEnd && frame.midi !== null,
    );
    const midi = median(inMora.map((frame) => frame.midi as number));
    const frequency = median(inMora.map((frame) => frame.frequency as number));
    const confidence = median(inMora.flatMap((frame) =>
      frame.confidence === undefined ? [] : [frame.confidence]));
    const source: NotePitchSource = inMora.some((frame) => frame.source === "swiftf0")
      ? "swiftf0"
      : inMora.some((frame) => frame.source === "yin")
        ? "yin"
        : inMora.some((frame) => frame.source === "interpolated") ? "interpolated" : "unvoiced";
    const mustRest = phoneme === "cl" || lyric === "っ";
    notes.push({
      start,
      end: Math.max(acousticEnd, start + 0.045),
      midi: mustRest || forceUnknownPitch || midi === null ? null : quantizeQuarterTone(midi),
      frequency: mustRest || forceUnknownPitch ? null : frequency,
      lyric,
      phoneme,
      pitchSource: mustRest || forceUnknownPitch ? "unvoiced" : source,
      confidence: mustRest || forceUnknownPitch ? null : confidence,
      notehead: forceUnknownPitch ? "x" : "normal",
    });
  };

  words.forEach(([wordStart, wordEnd, surface], wordIndex) => {
    const reading = wordReadings[wordIndex]?.surface === surface
      ? wordReadings[wordIndex]
      : wordReadings.find((item, index) => index >= wordIndex && index < wordIndex + 3 && item.surface === surface);
    const moras = reading?.moras.length ? reading.moras : splitJapaneseMoras(surface);
    if (!moras.length || wordEnd <= wordStart) return;

    const candidates = phones.flatMap(([start, end, phoneme], phoneIndex) => {
      const midpoint = (start + end) / 2;
      return SCORE_PHONES.has(phoneme) && midpoint >= wordStart - 0.015 && midpoint <= wordEnd + 0.015
        ? [{ start, end, phoneme, phoneIndex }]
        : [];
    });

    if (candidates.length === moras.length) {
      candidates.forEach((candidate, moraIndex) =>
        addNote(
          candidate.start,
          candidate.end,
          moras[moraIndex],
          candidate.phoneme,
          reading?.moraPitchSupported?.[moraIndex] === false,
        ));
      return;
    }

    // Forced alignment occasionally merges a rapid word (e.g. ラスト) into one
    // nucleus. Split its word interval so every sung mora remains one score note.
    const acousticWordEnd = Math.max(
      wordStart + moras.length * 0.045,
      trimTrailingSilence(wordStart, wordEnd),
    );
    const duration = acousticWordEnd - wordStart;
    moras.forEach((mora, moraIndex) => {
      const start = wordStart + (duration * moraIndex) / moras.length;
      const end = wordStart + (duration * (moraIndex + 1)) / moras.length;
      const midpoint = (start + end) / 2;
      const nearest = candidates.reduce<typeof candidates[number] | undefined>((best, candidate) => {
        if (!best) return candidate;
        const bestDistance = Math.abs((best.start + best.end) / 2 - midpoint);
        const distance = Math.abs((candidate.start + candidate.end) / 2 - midpoint);
        return distance < bestDistance ? candidate : best;
      }, undefined);
      const pitchSupported = reading?.moraPitchSupported?.[moraIndex] !== false;
      addNote(
        start,
        end,
        mora,
        mora === "っ" ? "cl" : pitchSupported ? nearest?.phoneme ?? "mora" : "mora",
        !pitchSupported,
      );
    });
  });

  const completeMissingPitches = (input: ScoreNote[]) => {
    const sorted = input.sort((a, b) => a.start - b.start);
    const anchors = sorted.flatMap((note, index) => note.midi === null ? [] : [{ note, index }]);
    const activeNotes = sorted.map((note) => activityInRange(note.start - 0.02, note.end + 0.02).isActive);
    const hasActivePath = (fromIndex: number, toIndex: number) => {
      const first = Math.min(fromIndex, toIndex);
      const last = Math.max(fromIndex, toIndex);
      for (let index = first; index <= last; index += 1) {
        if (!activeNotes[index]) return false;
        if (index > first && sorted[index].start - sorted[index - 1].end > 0.18) return false;
      }
      return true;
    };
    const completed = sorted.map((note, noteIndex) => {
      if (note.midi !== null || note.phoneme === "cl" || note.lyric === "っ") return note;
      const midpoint = (note.start + note.end) / 2;
      const before = anchors.findLast((anchor) => anchor.index < noteIndex);
      const after = anchors.find((anchor) => anchor.index > noteIndex);
      const beforeTime = before ? (before.note.start + before.note.end) / 2 : -Infinity;
      const afterTime = after ? (after.note.start + after.note.end) / 2 : Infinity;
      let estimatedMidi: number | null = null;

      if (before && after && afterTime - beforeTime <= 0.75) {
        const progress = Math.max(0, Math.min(1, (midpoint - beforeTime) / (afterTime - beforeTime)));
        estimatedMidi = (before.note.midi as number)
          + ((after.note.midi as number) - (before.note.midi as number)) * progress;
      } else if (before && midpoint - beforeTime <= 0.22) {
        estimatedMidi = before.note.midi;
      } else if (after && afterTime - midpoint <= 0.22) {
        estimatedMidi = after.note.midi;
      } else if (activeNotes[noteIndex]) {
        const activeBefore = before
          && midpoint - beforeTime <= 1.5
          && hasActivePath(before.index, noteIndex);
        const activeAfter = after
          && afterTime - midpoint <= 1.5
          && hasActivePath(noteIndex, after.index);
        if (activeBefore && activeAfter) {
          estimatedMidi = midpoint - beforeTime <= afterTime - midpoint ? before.note.midi : after.note.midi;
        } else if (activeBefore) {
          estimatedMidi = before.note.midi;
        } else if (activeAfter) {
          estimatedMidi = after.note.midi;
        }
      }

      if (estimatedMidi === null) return note;
      const roundedMidi = quantizeQuarterTone(estimatedMidi);
      return {
        ...note,
        midi: roundedMidi,
        frequency: 440 * 2 ** ((roundedMidi - 69) / 12),
        pitchSource: "interpolated" as const,
        confidence: null,
      };
    });

    const globalPitch = median(pitchFrames.flatMap((frame) => frame.midi === null ? [] : [frame.midi])) ?? 60;
    let previousPitch: number | null = null;
    return completed.map((note, noteIndex) => {
      if (note.midi !== null) {
        previousPitch = note.midi;
        return note;
      }
      if (note.phoneme === "cl" || note.lyric === "っ") return note;
      const nextPitch = completed.slice(noteIndex + 1).find((candidate) => candidate.midi !== null)?.midi;
      const carriedPitch = quantizeQuarterTone(previousPitch ?? nextPitch ?? globalPitch);
      previousPitch = carriedPitch;
      return {
        ...note,
        midi: carriedPitch,
        frequency: 440 * 2 ** ((carriedPitch - 69) / 12),
        pitchSource: "carried" as const,
        confidence: null,
        notehead: "x" as const,
      };
    });
  };

  if (notes.length) return completeMissingPitches(notes);

  // Fallback for aligners that return phones but no word spans.
  phones.forEach(([start, end, phoneme]) => {
    if (SCORE_PHONES.has(phoneme) && end - start >= 0.025) {
      addNote(start, end, phoneme === "cl" ? "っ" : phoneme, phoneme);
    }
  });
  return completeMissingPitches(notes);
}
