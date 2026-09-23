import type { ScoreNote } from "./analysis";

type TimedEvent = {
  start: number;
  duration: number;
  midi: number | null;
  lyric?: string;
  phoneme?: string;
  notehead?: "normal" | "x";
};

type NoteFragment = {
  event: TimedEvent;
  tieStart: boolean;
  tieStop: boolean;
};

type QuantizedNote = ScoreNote & {
  qStart: number;
  qEnd: number;
};

type MeasurePlan = {
  start: number;
  end: number;
  beats: number;
};

export type ScoreSettings = {
  quantum: number;
  minimumNoteLabel: string;
  clefLabel: string;
  clefXml: string;
  meterLabel: string;
  meters: number[];
};

const STEP_BY_PC = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"];
const ALTER_BY_PC = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0];
const DIVISIONS = 480;
const QUARTER = DIVISIONS;
const THIRTY_SECOND = DIVISIONS / 8;
const SIXTY_FOURTH = DIVISIONS / 16;
const NOTE_VALUES = [DIVISIONS * 4, DIVISIONS * 2, DIVISIONS, DIVISIONS / 2, DIVISIONS / 4, THIRTY_SECOND, SIXTY_FOURTH];
export const SCORE_BPM = 60;

function esc(text: string) {
  return text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function pitchXml(midi: number) {
  const semitone = Math.floor(midi);
  const pc = ((semitone % 12) + 12) % 12;
  const step = STEP_BY_PC[pc];
  const alter = ALTER_BY_PC[pc] + (midi - semitone);
  const octave = Math.floor(semitone / 12) - 1;
  return `<pitch><step>${step}</step>${alter ? `<alter>${alter}</alter>` : ""}<octave>${octave}</octave></pitch>`;
}

function accidentalXml(midi: number) {
  const semitone = Math.floor(midi);
  const fraction = midi - semitone;
  if (fraction < 0.25) return "";
  return ALTER_BY_PC[((semitone % 12) + 12) % 12]
    ? "<accidental>three-quarters-sharp</accidental>"
    : "<accidental>quarter-sharp</accidental>";
}

function typeForDuration(duration: number) {
  if (duration >= DIVISIONS * 4) return "whole";
  if (duration >= DIVISIONS * 2) return "half";
  if (duration >= DIVISIONS) return "quarter";
  if (duration >= DIVISIONS / 2) return "eighth";
  if (duration >= DIVISIONS / 4) return "16th";
  if (duration >= THIRTY_SECOND) return "32nd";
  return "64th";
}

function noteXml(event: TimedEvent, tieStart = false, tieStop = false, beams: string[] = []) {
  const restOrPitch = event.midi === null ? "<rest/>" : pitchXml(event.midi);
  const ties = `${tieStop ? '<tie type="stop"/>' : ""}${tieStart ? '<tie type="start"/>' : ""}`;
  const beamXml = beams.join("");
  const accidental = event.midi === null ? "" : accidentalXml(event.midi);
  const notehead = event.midi !== null && event.notehead === "x" ? "<notehead>x</notehead>" : "";
  const lyric = event.lyric
    ? `<lyric number="1"><syllabic>single</syllabic><text>${esc(event.lyric)}</text></lyric>`
    : "";
  const notationTies = tieStart || tieStop
    ? `<notations>${tieStop ? '<tied type="stop"/>' : ""}${tieStart ? '<tied type="start"/>' : ""}</notations>`
    : "";
  return `<note>${restOrPitch}${ties}<duration>${event.duration}</duration><voice>1</voice><type>${typeForDuration(event.duration)}</type>${accidental}${notehead}${beamXml}${notationTies}${lyric}</note>`;
}

function beamLevel(duration: number) {
  if (duration <= SIXTY_FOURTH) return 4;
  if (duration <= THIRTY_SECOND) return 3;
  if (duration <= DIVISIONS / 4) return 2;
  if (duration <= DIVISIONS / 2) return 1;
  return 0;
}

/**
 * MusicXML does not infer beams from note types. Group short notes and short
 * intervening rests inside each quarter-note beat, then explicitly mark every
 * primary/secondary beam. This keeps consonant gaps from breaking vocal beams.
 */
function renderBeamedFragments(fragments: NoteFragment[], measureStart: number) {
  const beamTags = fragments.map(() => [] as string[]);
  let runStart = -1;

  const closeRun = (runEnd: number) => {
    const pitchedCount = runStart < 0
      ? 0
      : fragments.slice(runStart, runEnd).filter((fragment) => fragment.event.midi !== null).length;
    if (runStart < 0 || runEnd - runStart < 2 || pitchedCount < 2) {
      runStart = -1;
      return;
    }
    for (let level = 1; level <= 4; level += 1) {
      for (let index = runStart; index < runEnd; index += 1) {
        if (beamLevel(fragments[index].event.duration) < level) continue;
        const hasPrevious = index > runStart && beamLevel(fragments[index - 1].event.duration) >= level;
        const hasNext = index + 1 < runEnd && beamLevel(fragments[index + 1].event.duration) >= level;
        let value: "begin" | "continue" | "end" | "forward hook" | "backward hook";
        if (hasPrevious && hasNext) value = "continue";
        else if (hasNext) value = "begin";
        else if (hasPrevious) value = "end";
        else value = index < runEnd - 1 ? "forward hook" : "backward hook";
        beamTags[index].push(`<beam number="${level}">${value}</beam>`);
      }
    }
    runStart = -1;
  };

  for (let index = 0; index <= fragments.length; index += 1) {
    const current = fragments[index];
    const previous = fragments[index - 1];
    const beamable = beamLevel(current?.event.duration ?? Infinity) > 0;
    const sameBeat = previous && current
      ? Math.floor((previous.event.start - measureStart) / QUARTER)
        === Math.floor((current.event.start - measureStart) / QUARTER)
      : false;
    const contiguous = previous && current
      ? previous.event.start + previous.event.duration === current.event.start
      : false;

    if (!beamable || (runStart >= 0 && (!sameBeat || !contiguous))) closeRun(index);
    if (beamable && runStart < 0) runStart = index;
  }

  return fragments.map((fragment, index) => noteXml(
    fragment.event,
    fragment.tieStart,
    fragment.tieStop,
    beamTags[index],
  )).join("");
}

/** Quantize to 1/64 without moving later sections away from the audio timeline. */
function quantizeTimeline(notes: ScoreNote[]) {
  const ticksPerSecond = DIVISIONS; // quarter = one second at 60 BPM
  const quantize = (seconds: number) => Math.max(
    0,
    Math.round((seconds * ticksPerSecond) / SIXTY_FOURTH) * SIXTY_FOURTH,
  );
  const raw = [...notes]
    .map((note) => {
      const qStart = quantize(note.start);
      return { ...note, qStart, qEnd: Math.max(qStart + SIXTY_FOURTH, quantize(note.end)) };
    })
    .sort((a, b) => a.qStart - b.qStart);

  const sectionAnchors = [0];
  for (let index = 1; index < raw.length; index += 1) {
    const note = raw[index];
    const previous = raw[index - 1];
    const gap = note.qStart - previous.qEnd;
    if (gap >= SIXTY_FOURTH * 2 && note.qStart % QUARTER === 0) sectionAnchors.push(note.qStart);
  }
  return { notes: raw, sectionAnchors };
}

function buildEvents(notes: QuantizedNote[], minimumTotalTicks = 0) {
  const events: TimedEvent[] = [];
  const onsets: number[] = [];
  let cursor = 0;
  for (const note of notes) {
    const start = Math.max(cursor, note.qStart);
    if (start > cursor) events.push({ start: cursor, duration: start - cursor, midi: null });
    const duration = Math.max(SIXTY_FOURTH, note.qEnd - start);
    events.push({ start, duration, midi: note.midi, lyric: note.lyric, phoneme: note.phoneme, notehead: note.notehead });
    onsets.push(start);
    cursor = start + duration;
  }
  if (!events.length) {
    events.push({ start: 0, duration: QUARTER * 4, midi: null });
    cursor = QUARTER * 4;
  }
  return { events, onsets, totalTicks: Math.max(cursor, minimumTotalTicks) };
}

function patternMismatch(patterns: Array<Set<number>>) {
  if (patterns.length < 2) return 0.8;
  let comparisons = 0;
  let mismatch = 0;
  for (let index = 1; index < patterns.length; index += 1) {
    const before = patterns[index - 1];
    const after = patterns[index];
    const union = new Set([...before, ...after]);
    if (!union.size) continue;
    let difference = 0;
    union.forEach((position) => {
      if (before.has(position) !== after.has(position)) difference += 1;
    });
    mismatch += difference / union.size;
    comparisons += 1;
  }
  return comparisons ? mismatch / comparisons : 0.8;
}

function isFeaturelessPulse(onsets: number[], start: number) {
  const relevant = onsets.filter((tick) => tick >= start && tick < start + QUARTER * 8);
  if (relevant.length < 4) return false;
  const intervals = relevant.slice(1).map((tick, index) => tick - relevant[index]);
  if (intervals.every((interval) => interval === intervals[0]) && intervals[0] <= QUARTER) return true;
  if (relevant.some((tick) => tick % QUARTER !== 0)) return false;
  const occupied = new Set(relevant.map((tick) => Math.round(tick / QUARTER)));
  const first = Math.min(...occupied);
  const last = Math.max(...occupied);
  for (let beat = first; beat <= last; beat += 1) if (!occupied.has(beat)) return false;
  return true;
}

function meterPatterns(start: number, beats: number, onsets: number[], totalTicks: number) {
  const cycleTicks = beats * QUARTER;
  const windowEnd = Math.min(totalTicks, start + Math.max(QUARTER * 12, cycleTicks * 4));
  const cycleCount = Math.floor((windowEnd - start) / cycleTicks);
  const patterns = Array.from({ length: cycleCount }, () => new Set<number>());
  onsets.forEach((tick) => {
    if (tick < start || tick >= start + cycleCount * cycleTicks) return;
    const cycle = Math.floor((tick - start) / cycleTicks);
    patterns[cycle].add(Math.round(((tick - start) % cycleTicks) / SIXTY_FOURTH));
  });
  return patterns;
}

function meterCost(start: number, beats: number, onsets: number[], totalTicks: number) {
  const patterns = meterPatterns(start, beats, onsets, totalTicks);
  if (patterns.length < 2) return 1 + beats * 0.02;
  const nonEmpty = patterns.filter((pattern) => pattern.size > 0);
  const downbeatSupport = nonEmpty.length
    ? nonEmpty.filter((pattern) => pattern.has(0)).length / nonEmpty.length
    : 0;
  const priors: Record<number, number> = { 1: 0.52, 2: 0.06, 3: 0.1, 4: 0.16 };
  return patternMismatch(patterns) * 0.78 + (1 - downbeatSupport) * 0.22 + priors[beats];
}

function inferMeterAt(start: number, onsets: number[], totalTicks: number) {
  const future = onsets.filter((tick) => tick >= start && tick < start + QUARTER * 12);
  if (future.length < 4 || isFeaturelessPulse(onsets, start)) return 4;
  return [1, 2, 3, 4]
    .map((beats) => ({ beats, cost: meterCost(start, beats, onsets, totalTicks) }))
    .sort((a, b) => a.cost - b.cost || b.beats - a.beats)[0].beats;
}

function hasStrongBoundary(tick: number, onsets: number[], sectionAnchors: number[]) {
  if (sectionAnchors.includes(tick)) return true;
  const index = onsets.indexOf(tick);
  if (index < 0) return false;
  const previous = onsets[index - 1];
  return previous === undefined || tick - previous >= QUARTER;
}

function buildMeasurePlan(totalTicks: number, onsets: number[], sectionAnchors: number[]) {
  const measures: MeasurePlan[] = [];
  let start = 0;
  let activeMeter = inferMeterAt(0, onsets, totalTicks);
  while (start < totalTicks) {
    const remainingBeats = Math.ceil((totalTicks - start) / QUARTER);
    if (remainingBeats <= activeMeter) {
      measures.push({ start, end: start + activeMeter * QUARTER, beats: activeMeter });
      break;
    }

    let shortenedTo = 0;
    let nextMeter = activeMeter;
    for (let beat = 1; beat < activeMeter; beat += 1) {
      const boundary = start + beat * QUARTER;
      if (!hasStrongBoundary(boundary, onsets, sectionAnchors)) continue;
      const candidate = inferMeterAt(boundary, onsets, totalTicks);
      const candidateCost = meterCost(boundary, candidate, onsets, totalTicks);
      const currentCost = meterCost(boundary, activeMeter, onsets, totalTicks);
      const explicitReanchor = sectionAnchors.includes(boundary);
      const coherentPattern = patternMismatch(meterPatterns(boundary, candidate, onsets, totalTicks).slice(0, 2)) <= 0.25;
      const enoughEvidence = totalTicks - boundary >= QUARTER * Math.max(activeMeter, candidate) * 2;
      if (coherentPattern && enoughEvidence
        && (explicitReanchor || (candidate !== activeMeter && candidateCost + 0.08 < currentCost))) {
        shortenedTo = beat;
        nextMeter = candidate;
        break;
      }
    }

    const beats = shortenedTo || activeMeter;
    const end = start + beats * QUARTER;
    measures.push({ start, end, beats });
    start = end;
    if (shortenedTo) {
      activeMeter = nextMeter;
      continue;
    }
    const candidate = inferMeterAt(start, onsets, totalTicks);
    if (candidate !== activeMeter && hasStrongBoundary(start, onsets, sectionAnchors)) {
      const stableCandidate = inferMeterAt(start + QUARTER, onsets, totalTicks);
      const evidenceTicks = QUARTER * Math.max(activeMeter, candidate) * 3;
      const coherentPattern = patternMismatch(meterPatterns(start, candidate, onsets, totalTicks).slice(0, 2)) <= 0.25;
      if (stableCandidate === candidate && coherentPattern && totalTicks - start >= evidenceTicks) activeMeter = candidate;
    }
  }
  return measures.length ? measures : [{ start: 0, end: QUARTER * 4, beats: 4 }];
}

function chooseClef(notes: ScoreNote[]) {
  const pitches = notes.flatMap((note) => note.midi === null ? [] : [note.midi]).sort((a, b) => a - b);
  const low = pitches.length ? pitches[Math.floor((pitches.length - 1) * 0.1)] : 60;
  const high = pitches.length ? pitches[Math.ceil((pitches.length - 1) * 0.9)] : 67;
  const rangeCenter = (low + high) / 2;
  return rangeCenter < 60.5
    ? { clefLabel: "ヘ音記号", clefXml: "<clef><sign>F</sign><line>4</line></clef>" }
    : { clefLabel: "ト音記号", clefXml: "<clef><sign>G</sign><line>2</line></clef>" };
}

function prepareScore(notes: ScoreNote[], totalDurationSeconds?: number) {
  const quantized = quantizeTimeline(notes);
  const durationTicks = totalDurationSeconds === undefined
    ? 0
    : Math.ceil((totalDurationSeconds * DIVISIONS) / SIXTY_FOURTH) * SIXTY_FOURTH;
  const timeline = buildEvents(quantized.notes, durationTicks);
  const measures = buildMeasurePlan(
    timeline.totalTicks,
    timeline.onsets,
    quantized.sectionAnchors,
  );
  return { ...timeline, measures };
}

export function inferScoreSettings(notes: ScoreNote[], totalDurationSeconds?: number): ScoreSettings {
  const score = prepareScore(notes, totalDurationSeconds);
  const meters = [...new Set(score.measures.map((measure) => measure.beats))];
  const meterLabel = meters.length > 1
    ? `可変拍子（${meters.map((beats) => `${beats}/4`).join("・")}）`
    : `${meters[0] ?? 4}/4`;
  return {
    quantum: SIXTY_FOURTH,
    minimumNoteLabel: "64分音符",
    ...chooseClef(notes),
    meterLabel,
    meters,
  };
}

export function makeMusicXml(notes: ScoreNote[], title = "Voice Score", totalDurationSeconds?: number) {
  const score = prepareScore(notes, totalDurationSeconds);
  const settings = inferScoreSettings(notes, totalDurationSeconds);
  let eventIndex = 0;
  let eventOffset = 0;
  let previousBeats = 0;
  const measures = score.measures.map((measure, measureIndex) => {
    const fragments: NoteFragment[] = [];
    let cursor = measure.start;
    while (cursor < measure.end) {
      const event = score.events[eventIndex];
      if (!event) {
        const remaining = measure.end - cursor;
        const part = NOTE_VALUES.find((candidate) => candidate <= remaining) ?? SIXTY_FOURTH;
        fragments.push({ event: { start: cursor, duration: part, midi: null }, tieStart: false, tieStop: false });
        cursor += part;
        continue;
      }
      if (cursor < event.start) {
        const gap = Math.min(measure.end - cursor, event.start - cursor);
        const part = NOTE_VALUES.find((candidate) => candidate <= gap) ?? SIXTY_FOURTH;
        fragments.push({ event: { start: cursor, duration: part, midi: null }, tieStart: false, tieStop: false });
        cursor += part;
        continue;
      }
      const remainingEvent = event.duration - eventOffset;
      const maximumPart = Math.min(remainingEvent, measure.end - cursor);
      const part = NOTE_VALUES.find((candidate) => candidate <= maximumPart) ?? SIXTY_FOURTH;
      const pitched = event.midi !== null;
      const tieStop = pitched && eventOffset > 0;
      const tieStart = pitched && eventOffset + part < event.duration;
      fragments.push({
        event: { ...event, start: cursor, duration: part, lyric: eventOffset === 0 ? event.lyric : undefined },
        tieStart,
        tieStop,
      });
      cursor += part;
      eventOffset += part;
      if (eventOffset >= event.duration) {
        eventIndex += 1;
        eventOffset = 0;
      }
    }
    const body = renderBeamedFragments(fragments, measure.start);

    const timeChanged = measureIndex === 0 || measure.beats !== previousBeats;
    const attributes = measureIndex === 0 || timeChanged
      ? `<attributes>${measureIndex === 0 ? `<divisions>${DIVISIONS}</divisions><key><fifths>0</fifths></key>` : ""}${timeChanged ? `<time><beats>${measure.beats}</beats><beat-type>4</beat-type></time>` : ""}${measureIndex === 0 ? settings.clefXml : ""}</attributes>`
      : "";
    const tempo = measureIndex === 0
      ? `<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${SCORE_BPM}</per-minute></metronome></direction-type><sound tempo="${SCORE_BPM}"/></direction>`
      : "";
    previousBeats = measure.beats;
    return `<measure number="${measureIndex + 1}">${attributes}${tempo}${body}</measure>`;
  });

  return `<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <work><work-title>${esc(title)}</work-title></work>
  <part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
  <part id="P1">${measures.join("")}</part>
</score-partwise>`;
}
