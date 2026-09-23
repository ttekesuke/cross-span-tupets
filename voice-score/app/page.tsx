"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { decodeToMono16k } from "charsiu-js/assets-web";
import type { OpenSheetMusicDisplay as OpenSheetMusicDisplayType } from "opensheetmusicdisplay";
import { midiToName, notesFromAlignment, type PitchFrame, type ScoreNote, type Segment, type WordReading } from "@/lib/analysis";
import { inferScoreSettings, makeMusicXml, SCORE_BPM } from "@/lib/musicxml";
import TranscribeWorker from "../workers/transcribe.worker.ts?worker";
import AnalysisWorker from "../workers/analysis.worker.ts?worker";

type Stage = "idle" | "decode" | "transcribe" | "align" | "pitch" | "score" | "done" | "error";

function formatClock(value: number) {
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

type AlignmentWithReadings = {
  phones: Segment[];
  words: Segment[];
  phoneIds: number[];
  wordReadings: WordReading[];
  omittedMoras: number;
};

type TranscriptChunk = { text: string; start: number; end: number };

function PitchPlot({ frames, phones }: { frames: PitchFrame[]; phones: Segment[] }) {
  const voiced = frames.filter((frame) => frame.midi !== null);
  const duration = Math.max(frames.at(-1)?.time ?? 1, phones.at(-1)?.[1] ?? 1);
  const minMidi = Math.floor(Math.min(...voiced.map((item) => item.midi as number), 55) - 2);
  const maxMidi = Math.ceil(Math.max(...voiced.map((item) => item.midi as number), 72) + 2);
  const x = (time: number) => 64 + (time / duration) * 1030;
  const y = (midi: number) => 22 + ((maxMidi - midi) / (maxMidi - minMidi)) * 154;
  let path = "";
  let penDown = false;
  for (const frame of frames) {
    if (frame.midi === null) { penDown = false; continue; }
    path += `${penDown ? "L" : "M"}${x(frame.time).toFixed(1)},${y(frame.midi).toFixed(1)} `;
    penDown = true;
  }
  return <svg viewBox="0 0 1120 224" className="pitch-plot" role="img" aria-label="時間に沿った検出音高と音素境界">
    <rect x="0" y="0" width="1120" height="224" fill="#fff" />
    {Array.from({ length: Math.max(1, Math.floor(duration) + 1) }, (_, index) => <g key={index}>
      <line x1={x(index)} x2={x(index)} y1="18" y2="180" stroke="#bbb" strokeDasharray="3 5" />
      <text x={x(index)} y="207" textAnchor="middle" fill="#555" fontSize="11">{index}s</text>
    </g>)}
    {[60, 64, 67, 72].filter((midi) => midi >= minMidi && midi <= maxMidi).map((midi) => <g key={midi}>
      <line x1="55" x2="1102" y1={y(midi)} y2={y(midi)} stroke="#ddd" />
      <text x="48" y={y(midi) + 4} textAnchor="end" fill="#555" fontSize="11">{midiToName(midi)}</text>
    </g>)}
    {phones.map(([start, end, phone], index) => <g key={`${start}-${phone}-${index}`}>
      <rect x={x(start)} y="182" width={Math.max(2, x(end) - x(start))} height="22" fill={index % 2 ? "#eee" : "#ddd"} />
      {x(end) - x(start) > 18 && <text x={(x(start) + x(end)) / 2} y="197" textAnchor="middle" fill="#222" fontSize="10">{phone}</text>}
    </g>)}
    <path d={path} fill="none" stroke="#000" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" />
  </svg>;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioDuration, setAudioDuration] = useState(0);
  const [audioRange, setAudioRange] = useState<[number, number]>([0, 0]);
  const [analyzedRange, setAnalyzedRange] = useState<[number, number] | null>(null);
  const [transcript, setTranscript] = useState("");
  const [transcriptIsAuto, setTranscriptIsAuto] = useState(false);
  const [phones, setPhones] = useState<Segment[]>([]);
  const [pitch, setPitch] = useState<PitchFrame[]>([]);
  const [notes, setNotes] = useState<ScoreNote[]>([]);
  const [stage, setStage] = useState<Stage>("idle");
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("音声を選択してください");
  const [error, setError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [isScorePlaying, setIsScorePlaying] = useState(false);
  const scoreRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const infoDialogRef = useRef<HTMLDialogElement>(null);
  const analysisDialogRef = useRef<HTMLDialogElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingStreamRef = useRef<MediaStream | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<number | null>(null);
  const recordingStartedAtRef = useRef(0);
  const osmdRef = useRef<OpenSheetMusicDisplayType | null>(null);
  const analysisWorkerRef = useRef<Worker | null>(null);
  const playbackFrameRef = useRef<number | null>(null);
  const lastCursorTimeRef = useRef(0);
  const scoreDuration = analyzedRange ? analyzedRange[1] - analyzedRange[0] : undefined;
  const musicXml = useMemo(
    () => makeMusicXml(notes, file?.name ?? "Voice Score", scoreDuration),
    [notes, file, scoreDuration],
  );
  const scoreSettings = useMemo(() => inferScoreSettings(notes, scoreDuration), [notes, scoreDuration]);

  useEffect(() => {
    let cancelled = false;
    async function renderScore() {
      if (!scoreRef.current) return;
      if (playbackFrameRef.current !== null) cancelAnimationFrame(playbackFrameRef.current);
      playbackFrameRef.current = null;
      audioRef.current?.pause();
      setIsScorePlaying(false);
      osmdRef.current = null;
      scoreRef.current.innerHTML = "";
      const { OpenSheetMusicDisplay } = await import("opensheetmusicdisplay");
      if (cancelled || !scoreRef.current) return;
      const osmd = new OpenSheetMusicDisplay(scoreRef.current, {
        autoResize: true,
        drawTitle: false,
        drawingParameters: "compacttight",
        backend: "svg",
        followCursor: true,
        cursorsOptions: [{ type: 0, color: "#555555", alpha: 0.65, follow: true }],
      });
      await osmd.load(musicXml);
      if (!cancelled) {
        osmd.render();
        osmd.cursor.reset();
        osmd.cursor.hide();
        osmdRef.current = osmd;
        lastCursorTimeRef.current = 0;
      }
    }
    renderScore().catch((reason) => setError(`楽譜表示に失敗しました: ${String(reason)}`));
    return () => { cancelled = true; };
  }, [musicXml]);

  useEffect(() => () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }, [audioUrl]);

  useEffect(() => () => {
    if (playbackFrameRef.current !== null) cancelAnimationFrame(playbackFrameRef.current);
    audioRef.current?.pause();
    if (recordingTimerRef.current !== null) window.clearInterval(recordingTimerRef.current);
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") { recorder.onstop = null; recorder.stop(); }
    recordingStreamRef.current?.getTracks().forEach((track) => track.stop());
    analysisWorkerRef.current?.terminate();
    analysisWorkerRef.current = null;
  }, []);

  function chooseFile(selected: File | null) {
    if (!selected) return;
    cancelPlaybackLoop();
    audioRef.current?.pause();
    setIsScorePlaying(false);
    setFile(selected);
    setAudioUrl(URL.createObjectURL(selected));
    setTranscript("");
    setTranscriptIsAuto(false);
    setAudioDuration(0);
    setAudioRange([0, 0]);
    setAnalyzedRange(null);
    setPhones([]);
    setPitch([]);
    setNotes([]);
    setStage("idle");
    setProgress(0);
    setError(null);
    setStatus("音声を読み込みました。解析範囲を選択できます");
  }

  function updateAudioRange(values: [number, number]) {
    setAudioRange(values);
    if (transcriptIsAuto) { setTranscript(""); setTranscriptIsAuto(false); }
  }

  function playSelectedRange() {
    if (!audioRef.current || audioRange[1] <= audioRange[0]) return;
    audioRef.current.currentTime = audioRange[0];
    void audioRef.current.play();
  }

  function cancelPlaybackLoop() {
    if (playbackFrameRef.current !== null) cancelAnimationFrame(playbackFrameRef.current);
    playbackFrameRef.current = null;
  }

  function syncScoreCursor(audioTime: number, forceReset = false) {
    const osmd = osmdRef.current;
    if (!osmd || !analyzedRange) return;
    const relativeSeconds = Math.max(0, audioTime - analyzedRange[0]);
    const cursor = osmd.cursor;
    if (forceReset || relativeSeconds + 0.03 < lastCursorTimeRef.current) {
      cursor.reset();
      cursor.show();
    } else if (cursor.Hidden) {
      cursor.show();
    }

    let steps = 0;
    while (!cursor.Iterator.EndReached && steps < 10_000) {
      const next = cursor.Iterator.clone();
      next.moveToNextVisibleVoiceEntry(false);
      if (next.EndReached) break;
      const nextSeconds = next.CurrentSourceTimestamp.RealValue * (240 / SCORE_BPM);
      if (nextSeconds > relativeSeconds + 0.015) break;
      cursor.next();
      steps += 1;
    }
    const viewport = scoreRef.current;
    const cursorElement = cursor.cursorElement;
    if (viewport && cursorElement) {
      const viewportRect = viewport.getBoundingClientRect();
      const cursorRect = cursorElement.getBoundingClientRect();
      if (cursorRect.left < viewportRect.left || cursorRect.right > viewportRect.right) {
        viewport.scrollTo({
          left: viewport.scrollLeft + cursorRect.left - viewportRect.left - viewportRect.width / 3,
          behavior: "smooth",
        });
      }
    }
    lastCursorTimeRef.current = relativeSeconds;
  }

  function beginPlaybackLoop() {
    const audio = audioRef.current;
    if (!audio || !analyzedRange) return;
    cancelPlaybackLoop();
    setIsScorePlaying(true);
    syncScoreCursor(audio.currentTime, true);

    const tick = () => {
      const activeAudio = audioRef.current;
      if (!activeAudio || activeAudio.paused) {
        cancelPlaybackLoop();
        setIsScorePlaying(false);
        return;
      }
      if (activeAudio.currentTime >= analyzedRange[1]) {
        stopScorePlayback();
        return;
      }
      syncScoreCursor(activeAudio.currentTime);
      playbackFrameRef.current = requestAnimationFrame(tick);
    };
    playbackFrameRef.current = requestAnimationFrame(tick);
  }

  async function playScore() {
    const audio = audioRef.current;
    if (!audio || !analyzedRange) return;
    if (audio.currentTime < analyzedRange[0] || audio.currentTime >= analyzedRange[1]) {
      audio.currentTime = analyzedRange[0];
    }
    syncScoreCursor(audio.currentTime, true);
    try {
      await audio.play();
      if (playbackFrameRef.current === null) beginPlaybackLoop();
    } catch (reason) {
      setError(`音声を再生できませんでした: ${String(reason)}`);
    }
  }

  function stopScorePlayback() {
    cancelPlaybackLoop();
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      if (analyzedRange) audio.currentTime = analyzedRange[0];
    }
    setIsScorePlaying(false);
    if (analyzedRange) syncScoreCursor(analyzedRange[0], true);
  }

  function handleAudioPlay() {
    const audio = audioRef.current;
    if (!audio || !analyzedRange) return;
    if (audio.currentTime >= analyzedRange[0] && audio.currentTime < analyzedRange[1]) beginPlaybackLoop();
  }

  function handleAudioPause() {
    cancelPlaybackLoop();
    setIsScorePlaying(false);
  }

  async function startRecording() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("このブラウザは音声録音に対応していません");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"];
      const mimeType = candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      recordedChunksRef.current = [];
      recordingStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => { if (event.data.size > 0) recordedChunksRef.current.push(event.data); };
      recorder.onerror = () => setError("録音中にエラーが発生しました");
      recorder.onstop = () => {
        if (recordingTimerRef.current !== null) window.clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
        stream.getTracks().forEach((track) => track.stop());
        recordingStreamRef.current = null;
        mediaRecorderRef.current = null;
        setIsRecording(false);
        const chunks = recordedChunksRef.current;
        if (!chunks.length) { setError("録音データを取得できませんでした"); return; }
        const recordedType = recorder.mimeType || mimeType || "audio/webm";
        const extension = recordedType.includes("ogg") ? "ogg" : recordedType.includes("mp4") ? "m4a" : "webm";
        chooseFile(new File(chunks, `recording-${new Date().toISOString().replace(/[:.]/g, "-")}.${extension}`, { type: recordedType }));
      };
      recorder.start(250);
      recordingStartedAtRef.current = performance.now();
      setRecordingSeconds(0);
      setIsRecording(true);
      setStatus("録音中です");
      recordingTimerRef.current = window.setInterval(() => setRecordingSeconds((performance.now() - recordingStartedAtRef.current) / 1000), 250);
    } catch (reason) {
      const name = reason instanceof DOMException ? reason.name : "";
      setError(name === "NotAllowedError" ? "マイクの使用が許可されませんでした" : `録音を開始できませんでした: ${String(reason)}`);
    }
  }

  function stopRecording() {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }

  async function transcribe(raw: Float32Array) {
    setStage("transcribe");
    setStatus("Whisper をバックグラウンドで読み込んでいます（初回はモデルを取得します）");
    return new Promise<{ text: string; chunks: TranscriptChunk[] }>((resolve, reject) => {
      const worker = new TranscribeWorker();
      const timeout = window.setTimeout(() => { worker.terminate(); reject(new Error("文字起こしが15分以内に完了しませんでした。解析範囲を短くして再実行してください")); }, 15 * 60_000);
      const finish = () => { window.clearTimeout(timeout); worker.terminate(); };
      worker.onmessage = (event: MessageEvent<{ type: string; text?: string; chunks?: TranscriptChunk[]; message?: string; percent?: number }>) => {
        if (event.data.type === "progress" && event.data.message) {
          if (event.data.percent !== undefined) setProgress((current) => Math.max(current, event.data.percent ?? current));
          setStatus(event.data.message);
        } else if (event.data.type === "result") {
          setProgress((current) => Math.max(current, event.data.percent ?? 45));
          finish();
          resolve({ text: event.data.text ?? "", chunks: event.data.chunks ?? [] });
        } else if (event.data.type === "error") { finish(); reject(new Error(event.data.message ?? "文字起こしに失敗しました")); }
      };
      worker.onerror = (event) => { finish(); reject(new Error(event.message || "文字起こし用のバックグラウンド処理が停止しました")); };
      const workerAudio = raw.slice();
      worker.postMessage({ type: "transcribe", audio: workerAudio }, [workerAudio.buffer]);
    });
  }

  async function analyzeInBackground(raw: Float32Array, text: string) {
    setStage("align");
    return new Promise<{ aligned: AlignmentWithReadings; pitchFrames: PitchFrame[] }>((resolve, reject) => {
      const worker = analysisWorkerRef.current ?? new AnalysisWorker();
      analysisWorkerRef.current = worker;
      const timeout = window.setTimeout(() => {
        worker.terminate();
        if (analysisWorkerRef.current === worker) analysisWorkerRef.current = null;
        reject(new Error("音素・音高解析が20分以内に完了しませんでした。解析範囲を短くして再実行してください"));
      }, 20 * 60_000);
      const finish = () => {
        window.clearTimeout(timeout);
        worker.onmessage = null;
        worker.onerror = null;
      };
      worker.onmessage = (event: MessageEvent<{ type: string; stage?: "align" | "pitch"; percent?: number; message?: string; aligned?: AlignmentWithReadings; pitchFrames?: PitchFrame[] }>) => {
        if (event.data.type === "progress") {
          if (event.data.stage) setStage(event.data.stage);
          if (event.data.percent !== undefined) setProgress((current) => Math.max(current, event.data.percent ?? current));
          if (event.data.message) setStatus(event.data.message);
        } else if (event.data.type === "result" && event.data.aligned && event.data.pitchFrames) {
          setProgress((current) => Math.max(current, event.data.percent ?? 96));
          finish();
          resolve({ aligned: event.data.aligned, pitchFrames: event.data.pitchFrames });
        } else if (event.data.type === "error") { finish(); reject(new Error(event.data.message ?? "音素・音高解析に失敗しました")); }
      };
      worker.onerror = (event) => {
        finish();
        worker.terminate();
        if (analysisWorkerRef.current === worker) analysisWorkerRef.current = null;
        reject(new Error(event.message || "音素・音高解析のバックグラウンド処理が停止しました"));
      };
      worker.postMessage({ type: "analyze", audio: raw, text }, [raw.buffer]);
    });
  }

  async function runAnalysis() {
    if (!file) { fileInputRef.current?.click(); return; }
    setError(null);
    try {
      setProgress(2); setStage("decode"); setStatus("選択区間を 16 kHz・モノラルへ変換しています");
      const decoded = await decodeToMono16k(await file.arrayBuffer());
      const decodedDuration = decoded.length / 16000;
      const rangeStart = audioDuration > 0 ? Math.max(0, Math.min(audioRange[0], decodedDuration)) : 0;
      const rangeEnd = audioDuration > 0 ? Math.max(rangeStart, Math.min(audioRange[1], decodedDuration)) : decodedDuration;
      if (rangeEnd - rangeStart < 0.25) throw new Error("解析範囲は0.25秒以上にしてください");
      const raw = decoded.slice(Math.floor(rangeStart * 16000), Math.ceil(rangeEnd * 16000));
      setProgress(8);
      let recognized = transcript.trim();
      if (!recognized) {
        const result = await transcribe(raw);
        recognized = result.text;
        setTranscript(recognized); setTranscriptIsAuto(true);
      } else { setProgress(45); }
      if (!recognized) throw new Error("文字起こし結果が空でした。歌詞欄へ手入力して再実行してください");
      await new Promise((resolve) => window.setTimeout(resolve, 80));
      const { aligned, pitchFrames } = await analyzeInBackground(raw, recognized);
      setPhones(aligned.phones); setPitch(pitchFrames);
      setProgress(97); setStage("score"); setStatus("母音・撥音の時刻へ音高を対応させ、MusicXML を生成しています");
      const scoreNotes = notesFromAlignment(aligned.phones, aligned.words, pitchFrames, aligned.wordReadings);
      if (!scoreNotes.length) throw new Error("有効な音高が検出できませんでした。単旋律の声だけの音源を試してください");
      setNotes(scoreNotes); setAnalyzedRange([rangeStart, rangeEnd]); setProgress(100); setStage("done");
      const estimatedCount = scoreNotes.filter((note) => note.pitchSource === "interpolated" && note.notehead !== "x").length;
      const unknownPitchCount = scoreNotes.filter((note) => note.notehead === "x").length;
      const details = [
        estimatedCount ? `${estimatedCount}音は前後から補間` : "",
        unknownPitchCount ? `${unknownPitchCount}音は音高不明の×符頭` : "",
        aligned.omittedMoras ? `音響上見つからない${aligned.omittedMoras}モーラは省略` : "",
      ].filter(Boolean).join("、");
      setStatus(`${formatClock(rangeStart)}–${formatClock(rangeEnd)} から ${scoreNotes.length}音を採譜しました${details ? `（${details}）` : ""}`);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason);
      setStage("error"); setError(message); setStatus("解析を完了できませんでした");
    }
  }

  function downloadMusicXml() {
    const blob = new Blob([musicXml], { type: "application/vnd.recordare.musicxml+xml" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${file?.name.replace(/\.[^.]+$/, "") || "voice-score"}.musicxml`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function downloadCsv() {
    const escapeCell = (value: string | number) => {
      const text = String(value);
      return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    };
    const offset = analyzedRange?.[0] ?? 0;
    const header = ["歌詞", "音素", "開始時刻秒", "終了時刻秒", "音名", "MIDI", "周波数Hz", "符頭", "検出方式", "信頼度"];
    const rows = notes.map((note) => [note.lyric, note.phoneme, (note.start + offset).toFixed(3), (note.end + offset).toFixed(3), note.midi === null ? "休符" : midiToName(note.midi), note.midi ?? "", note.frequency === null ? "" : note.frequency.toFixed(2), note.notehead === "x" ? "×" : "通常", note.pitchSource, note.confidence === null ? "" : note.confidence.toFixed(3)]);
    const csv = [header, ...rows].map((row) => row.map(escapeCell).join(",")).join("\r\n");
    const blob = new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${file?.name.replace(/\.[^.]+$/, "") || "voice-score"}-notes.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const isBusy = !["idle", "done", "error"].includes(stage);
  const voicedCount = pitch.filter((item) => item.midi !== null).length;
  const pitchCoverage = pitch.length ? Math.round((voicedCount / pitch.length) * 100) : 0;
  const soundingNotes = notes.flatMap((note) => note.midi === null ? [] : [note.midi]);
  const detectedRange = soundingNotes.length ? `${midiToName(Math.min(...soundingNotes))}–${midiToName(Math.max(...soundingNotes))}` : "—";
  const playbackAvailable = Boolean(audioUrl && analyzedRange);

  return <main className="app-shell">
    <div className="page-title-row">
      <h1>音声楽譜化</h1>
      <button type="button" onClick={() => infoDialogRef.current?.showModal()}>説明</button>
    </div>

    <dialog ref={infoDialogRef}>
      <h2>説明</h2>
      <p>音声ファイルまたはブラウザ録音から、歌詞付きの楽譜を生成します。処理はブラウザ内で行います。</p>
      <ul>
        <li>日本語専用です。</li>
        <li>歌声の解析は難しいため、現在は歌ではなく発話音声を推奨します。</li>
        <li>拍子の自動配置はまだ十分な精度ではありません。</li>
        <li>入力した歌詞を正解として音声へ当て込みます。自動文字起こしの正確さは低いため、正しい歌詞の入力を推奨します。</li>
        <li>歌詞は漢字を避け、実際の発音どおりのひらがなまたはカタカナに統一すると、複数の読みや固有名詞による読み間違いを防げます。</li>
        <li>音の高さを判定できなかった文字は、直前の音高を引き継いだ×符頭で表示します。</li>
        <li>解析範囲は最大3分程度を推奨します。</li>
      </ul>
      <dl>
        <dt>文字起こし</dt><dd>歌詞が空の場合だけSilero VADとWhisperで日本語認識。手入力歌詞は変更せず、Whisperを通さない</dd>
        <dt>音素境界</dt><dd>音声全体のHuBERT出力を重複区間で計算・連結し、低メモリで全文を単調整列。Whisperの推定時刻は使用しない。※モデルは初回解析時にロードしてメモリに保持し、同じページを開いている間は2回目以降もそれを使用します</dd>
        <dt>歌詞の欠落</dt><dd>歌詞の順序を保ちながら、音響上見つからないモーラは無理に音符へせず省略し、次に一致する発音から同期を戻す</dd>
        <dt>音高</dt><dd>SwiftF0 を主に使用し、YIN・音響活動・前後の実測値で補助。四分音単位で記譜</dd>
        <dt>音高不明</dt><dd>歌詞と発音位置があるのに音高を検出できない場合は、直前の音高を引き継ぎ、×符頭で記譜</dd>
        <dt>採譜</dt><dd>60 BPM、最小単位1/64音符、1/4〜4/4の範囲で拍子を自動推定</dd>
        <dt>音部記号</dt><dd>解析音域に応じてト音記号またはヘ音記号を選択</dd>
        <dt>音価</dt><dd>音素境界の後ろに無音が含まれる場合は、実際の音響が終わった位置で音符を切って休符化</dd>
        <dt>再生</dt><dd>元音声の再生時刻と楽譜カーソルを同期</dd>
        <dt>出力</dt><dd>MusicXML、音素と音高のCSV</dd>
      </dl>
      <form method="dialog"><button>閉じる</button></form>
    </dialog>

    <dialog ref={analysisDialogRef} className="analysis-dialog">
      <div className="section-heading">
        <div><h2>音素と音高</h2><p>解析結果</p></div>
        <button type="button" onClick={downloadCsv}>CSVダウンロード</button>
      </div>
      <PitchPlot frames={pitch} phones={phones} />
      <dl className="metrics">
        <div><dt>音素</dt><dd>{phones.filter((item) => item[2] !== "[SIL]").length}</dd></div>
        <div><dt>採譜音</dt><dd>{notes.length}</dd></div>
        <div><dt>有声音率</dt><dd>{pitchCoverage}%</dd></div>
        <div><dt>音域</dt><dd>{detectedRange}</dd></div>
      </dl>
      <form method="dialog"><button>閉じる</button></form>
    </dialog>

    <section className="workspace-grid">
      <aside className="control-panel">
        <h2>入力音声</h2>
        <div className="input-block">
          <label htmlFor="audio-file">音声ファイル</label>
          <input ref={fileInputRef} id="audio-file" type="file" accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm" disabled={isBusy || isRecording} onChange={(event) => chooseFile(event.target.files?.[0] ?? null)} />
          {file && <span>{file.name}（{(file.size / 1024 / 1024).toFixed(1)} MB）</span>}
        </div>
        <div className="recording-controls">
          {isRecording ? <button type="button" onClick={stopRecording}>録音停止（{formatClock(recordingSeconds)}）</button> : <button type="button" onClick={startRecording} disabled={isBusy}>マイクで録音</button>}
        </div>

        {audioUrl && <>
          <audio ref={audioRef} className="audio-player" src={audioUrl} controls preload="metadata"
            onLoadedMetadata={(event) => { const duration = Number.isFinite(event.currentTarget.duration) ? event.currentTarget.duration : 0; setAudioDuration(duration); setAudioRange([0, duration]); }}
            onPlay={handleAudioPlay}
            onPause={handleAudioPause}
            onSeeked={(event) => {
              if (analyzedRange && event.currentTarget.currentTime >= analyzedRange[0] && event.currentTarget.currentTime < analyzedRange[1]) {
                syncScoreCursor(event.currentTarget.currentTime, true);
              }
            }}
            onTimeUpdate={(event) => { if (event.currentTarget.currentTime >= audioRange[1]) event.currentTarget.pause(); }} />
          {audioDuration > 0 && <fieldset className="range-editor" disabled={isBusy}>
            <legend>解析範囲</legend>
            <output>{formatClock(audioRange[0])} – {formatClock(audioRange[1])}（{formatClock(audioRange[1] - audioRange[0])}）</output>
            <label htmlFor="range-start">開始</label>
            <input id="range-start" type="range" min={0} max={Math.max(0, audioRange[1] - 0.3)} step={0.1} value={audioRange[0]} onChange={(event) => updateAudioRange([Number(event.target.value), audioRange[1]])} />
            <label htmlFor="range-end">終了</label>
            <input id="range-end" type="range" min={Math.min(audioDuration, audioRange[0] + 0.3)} max={audioDuration} step={0.1} value={audioRange[1]} onChange={(event) => updateAudioRange([audioRange[0], Number(event.target.value)])} />
            <button type="button" onClick={playSelectedRange}>選択区間を再生</button>
          </fieldset>}
        </>}

        <div className="input-block">
          <label htmlFor="transcript">認識歌詞（空欄なら自動文字起こし）</label>
          <textarea id="transcript" value={transcript} onChange={(event) => { setTranscript(event.target.value); setTranscriptIsAuto(false); }} placeholder="正しい歌詞を、実際の発音どおりのひらがな／カタカナで入力することを推奨します。" rows={7} disabled={isBusy} />
        </div>
        <p>テンポ: {SCORE_BPM} BPM / 拍子: 自動</p>
        <div className="analysis-controls">
          <button type="button" onClick={runAnalysis} disabled={isBusy || isRecording}>{isBusy ? `解析中 ${progress}%` : file ? "選択区間を楽譜化" : "音声を選んで解析"}</button>
        </div>
        <section aria-live="polite" className="status-section">
          <h3>{error ? "エラー" : stage === "done" ? "解析完了" : "状態"}</h3>
          <p>
            {error ?? status}
            {stage === "done" && <> <button type="button" onClick={() => analysisDialogRef.current?.showModal()}>音素と音高</button></>}
          </p>
          {isBusy && <progress value={progress} max={100}>{progress}%</progress>}
        </section>
      </aside>

      <div className="results-panel">
        <section className="result-section score-section">
          <div className="section-heading">
            <div><h2>歌詞付き楽譜</h2><p>{scoreSettings.meterLabel} / {SCORE_BPM} BPM / 最小音価 {scoreSettings.minimumNoteLabel} / {scoreSettings.clefLabel}</p></div>
            <div className="score-actions">
              <button type="button" onClick={playScore} disabled={!playbackAvailable || isScorePlaying}>再生</button>
              <button type="button" onClick={stopScorePlayback} disabled={!playbackAvailable}>停止</button>
              <button type="button" onClick={downloadMusicXml}>MusicXMLダウンロード</button>
            </div>
          </div>
          <div className="score-paper" ref={scoreRef} />
        </section>
      </div>
    </section>
  </main>;
}
