(() => {
'use strict';

const ANALYSIS_CELL_PX = 5;
const ANALYSIS_COLS_PER_SCORE_COL = 4;
const PITCHES_PER_STAFF_FIXED = 8;
const COLS_PER_MEASURE = 4;
const MEASURE_WIDTH = 160;
const STAFF_GAP = 40;
const STAFF_BAND = 80;
const TOP_MARGIN = 60;
const BOTTOM_MARGIN = 60;
const NATURAL_MIDIS = [83,79,76,72,69,65,62,59];

function buildPitchTable(staffCount) {
    return Array.from({length:staffCount},()=>NATURAL_MIDIS.slice());
}

function buildLayout(srcW, srcH) {
    const analysisCols=Math.max(4,Math.round(srcW/ANALYSIS_CELL_PX));
    const analysisRows=Math.max(PITCHES_PER_STAFF_FIXED,Math.round(srcH/ANALYSIS_CELL_PX));
    const scoreColsUsed=Math.max(1,Math.ceil(analysisCols/ANALYSIS_COLS_PER_SCORE_COL));
    const staffCount=Math.max(1,Math.ceil(analysisRows/PITCHES_PER_STAFF_FIXED));
    const measureCount=Math.max(1,Math.ceil(scoreColsUsed/COLS_PER_MEASURE));
    return {srcW,srcH,analysisCols,analysisRows,scoreColsUsed,scoreColsPadded:measureCount*COLS_PER_MEASURE,analysisColsPerScoreCol:ANALYSIS_COLS_PER_SCORE_COL,staffCount,pitchesPerStaff:PITCHES_PER_STAFF_FIXED,measureCount,colsPerMeasure:COLS_PER_MEASURE,measureWidth:MEASURE_WIDTH,staffGap:STAFF_GAP,staffBand:STAFF_BAND,topMargin:TOP_MARGIN,bottomMargin:BOTTOM_MARGIN,scoreWidth:measureCount*MEASURE_WIDTH,scoreHeight:TOP_MARGIN+staffCount*STAFF_BAND+BOTTOM_MARGIN};
}

function convert(rawTraces, srcW, srcH, segC) {
            if (!segC) return null;

            const layout = buildLayout(srcW, srcH);
            const HD_W = srcW, HD_H = srcH;
            const ANALYSIS_GRID_X = layout.analysisCols;
            const ANALYSIS_GRID_Y = layout.analysisRows;
            const SCORE_GRID_X = layout.scoreColsUsed;
            const NUM_STAVES = layout.staffCount;
            const PITCHES_PER_STAFF = layout.pitchesPerStaff;
            const PITCHES_PER_STAVE = PITCHES_PER_STAFF;
            const ANALYSIS_COLS_PER_SCORE_COL = layout.analysisColsPerScoreCol;
            const PITCH_TABLE = buildPitchTable(NUM_STAVES);
            const cellW = HD_W / ANALYSIS_GRID_X;
            const pitchH = HD_H / ANALYSIS_GRID_Y;

            function clampInt(v, lo, hi) {
                return Math.max(lo, Math.min(hi, v));
            }

            function rasterizeGridLine(x0, y0, x1, y1) {
                const pts = [];
                let cx = x0, cy = y0;
                const dx = Math.abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
                const dy = -Math.abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
                let err = dx + dy;
                while (true) {
                    pts.push({ x: cx, gy: cy });
                    if (cx === x1 && cy === y1) break;
                    const e2 = 2 * err;
                    if (e2 >= dy) { err += dy; cx += sx; }
                    if (e2 <= dx) { err += dx; cy += sy; }
                }
                return pts;
            }

            const rawSegments = [];
            for (const pts of rawTraces) {
                if (!pts || pts.length < 2) continue;

                const quantized = [];
                for (const p of pts) {
                    const qx = clampInt(Math.floor(p.x / cellW), 0, ANALYSIS_GRID_X - 1);
                    const qy = clampInt(Math.round(p.y / pitchH), 0, ANALYSIS_GRID_Y - 1);
                    const last = quantized[quantized.length - 1];
                    if (!last || last.x !== qx || last.gy !== qy) {
                        quantized.push({ x: qx, gy: qy });
                    }
                }
                if (quantized.length < 2) continue;

                const path = [];
                for (let i = 1; i < quantized.length; i++) {
                    const a = quantized[i - 1];
                    const b = quantized[i];
                    const sub = rasterizeGridLine(a.x, a.gy, b.x, b.gy);
                    for (let j = 0; j < sub.length; j++) {
                        if (path.length > 0 && j === 0) {
                            const last = path[path.length - 1];
                            if (last.x === sub[j].x && last.gy === sub[j].gy) continue;
                        }
                        path.push(sub[j]);
                    }
                }
                if (path.length < 2) continue;

                const cleaned = [];
                for (const p of path) {
                    if (cleaned.length >= 2) {
                        const a = cleaned[cleaned.length - 2];
                        const b = cleaned[cleaned.length - 1];
                        if (a.x === p.x && a.gy === p.gy && !(b.x === p.x && b.gy === p.gy)) {
                            cleaned.pop();
                            continue;
                        }
                    }
                    const last = cleaned[cleaned.length - 1];
                    if (!last || last.x !== p.x || last.gy !== p.gy) cleaned.push(p);
                }
                if (cleaned.length < 2) continue;

                const interiorFragments = [];
                let frag = [];
                const flushFrag = () => {
                    if (frag.length >= 2) interiorFragments.push(frag);
                    frag = [];
                };
                for (const p of cleaned) {
                    const onGridBorder = (
                        p.x <= 0 || p.x >= ANALYSIS_GRID_X - 1 ||
                        p.gy <= 0 || p.gy >= ANALYSIS_GRID_Y - 1
                    );
                    if (onGridBorder) {
                        flushFrag();
                        continue;
                    }
                    const last = frag[frag.length - 1];
                    if (!last || last.x !== p.x || last.gy !== p.gy) frag.push(p);
                }
                flushFrag();

                for (const interior of interiorFragments) {
                    let run = [interior[0]];
                    let dir = 0;
                    const flushRun = () => {
                        if (run.length < 2) return;
                        // 垂直線も捨てない。以前は同一xの2点だけのrunを除外していたが、
                        // Geometry modeではそれ自体が意味のある垂直グリッサンドになる。
                        rawSegments.push(dir === -1 ? run.slice().reverse() : run.slice());
                    };

                    for (let i = 1; i < interior.length; i++) {
                        const prev = interior[i - 1];
                        const cur = interior[i];
                        const dx = cur.x - prev.x;
                        if (dx === 0) {
                            run.push(cur);
                            continue;
                        }
                        const curDir = dx > 0 ? 1 : -1;
                        if (dir === 0) {
                            dir = curDir;
                            run.push(cur);
                        } else if (curDir === dir) {
                            run.push(cur);
                        } else {
                            flushRun();
                            run = [prev, cur];
                            dir = curDir;
                        }
                    }
                    flushRun();
                }
            }

            const segments = [];
            for (const raw of rawSegments) {
                let run = [];
                let currentStaff = null;
                let prevPoint = null;

                for (const p of raw) {
                    const staff = clampInt(Math.floor(p.gy / PITCHES_PER_STAFF), 0, NUM_STAVES - 1);
                    const pitch = clampInt(p.gy - staff * PITCHES_PER_STAFF, 0, PITCHES_PER_STAFF - 1);
                    const point = { x: p.x, pitch };

                    if (currentStaff === null) {
                        currentStaff = staff;
                        run = [point];
                    } else if (staff === currentStaff) {
                        const last = run[run.length - 1];
                        if (!last || last.x !== point.x || last.pitch !== point.pitch) run.push(point);
                    } else {
                        if (prevPoint) {
                            const bridgePrevPitch = clampInt(prevPoint.gy - currentStaff * PITCHES_PER_STAFF, 0, PITCHES_PER_STAFF - 1);
                            const last = run[run.length - 1];
                            if (!last || last.x !== prevPoint.x || last.pitch !== bridgePrevPitch) {
                                run.push({ x: prevPoint.x, pitch: bridgePrevPitch });
                            }
                        }
                        if (run.length >= 2) segments.push({ staff: currentStaff, points: run });

                        currentStaff = staff;
                        const startRun = [];
                        if (prevPoint) {
                            const bridgeCurrPitch = clampInt(prevPoint.gy - currentStaff * PITCHES_PER_STAFF, 0, PITCHES_PER_STAFF - 1);
                            startRun.push({ x: prevPoint.x, pitch: bridgeCurrPitch });
                        }
                        startRun.push(point);
                        run = startRun;
                    }
                    prevPoint = p;
                }

                if (run.length >= 2) {
                    segments.push({ staff: currentStaff, points: run });
                }
            }

            const segCtx = segC.getContext('2d');
            segCtx.clearRect(0, 0, HD_W, HD_H);
            segCtx.strokeStyle = '#ff0000';
            segCtx.lineWidth = 4;
            segCtx.lineJoin = 'round';
            segCtx.lineCap = 'round';
            for (const seg of segments) {
                if (!seg.points || seg.points.length < 2) continue;
                const first = seg.points[0];
                segCtx.beginPath();
                segCtx.moveTo(
                    first.x * cellW + cellW / 2,
                    (seg.staff * PITCHES_PER_STAFF + first.pitch + 0.5) * pitchH
                );
                for (let i = 1; i < seg.points.length; i++) {
                    const p = seg.points[i];
                    segCtx.lineTo(
                        p.x * cellW + cellW / 2,
                        (seg.staff * PITCHES_PER_STAFF + p.pitch + 0.5) * pitchH
                    );
                }
                segCtx.stroke();
            }

            const scoreSegments = [];
            for (const seg of segments) {
                if (!seg.points || seg.points.length < 2) continue;

                // 解析4列 -> 四分音符1列へ縮約する。ただし、実際に解析グリッド上で
                // xが同じ垂直線だけは上下2点を残す。単に傾きが急な斜線を垂直扱い
                // すると、通常の大きな跳躍まで短い急線に置き換わってしまう。
                const buckets = new Map();
                for (const p of seg.points) {
                    const sx = clampInt(
                        Math.floor(p.x / ANALYSIS_COLS_PER_SCORE_COL),
                        0,
                        SCORE_GRID_X - 1
                    );
                    if (!buckets.has(sx)) buckets.set(sx, []);
                    buckets.get(sx).push(p);
                }

                const scorePoints = [];
                const sortedXs = Array.from(buckets.keys()).sort((a, b) => a - b);
                for (const sx of sortedXs) {
                    const candidates = buckets.get(sx);
                    const xs = candidates.map(p => p.x);
                    const pitches = candidates.map(p => p.pitch);
                    const analysisXSpan = Math.max(...xs) - Math.min(...xs);
                    const pitchMin = Math.min(...pitches);
                    const pitchMax = Math.max(...pitches);
                    const pitchSpan = pitchMax - pitchMin;

                    const verticalish = pitchSpan >= 1 && analysisXSpan === 0;

                    if (verticalish) {
                        const firstPitch = candidates[0].pitch;
                        const lastPitch = candidates[candidates.length - 1].pitch;
                        let aPitch = firstPitch;
                        let bPitch = lastPitch;
                        if (aPitch === bPitch) {
                            // 同じ高さへ戻るU字状の線でも、縦方向の振れを1本残す。
                            bPitch = Math.abs(pitchMax - aPitch) >= Math.abs(pitchMin - aPitch)
                                ? pitchMax : pitchMin;
                        }
                        if (aPitch !== bPitch) {
                            scorePoints.push({ x: sx, pitch: aPitch });
                            scorePoints.push({ x: sx, pitch: bPitch });
                            continue;
                        }
                    }

                    const centerAnalysisX = sx * ANALYSIS_COLS_PER_SCORE_COL
                        + (ANALYSIS_COLS_PER_SCORE_COL - 1) / 2;
                    let best = candidates[0];
                    let bestDist = Math.abs(best.x - centerAnalysisX);
                    for (let i = 1; i < candidates.length; i++) {
                        const d = Math.abs(candidates[i].x - centerAnalysisX);
                        if (d < bestDist) {
                            best = candidates[i];
                            bestDist = d;
                        }
                    }
                    scorePoints.push({ x: sx, pitch: best.pitch });
                }

                if (scorePoints.length >= 2) {
                    scoreSegments.push({ staff: seg.staff, points: scorePoints });
                }
            }

            const voiceEndsByStaff = Array.from({ length: NUM_STAVES }, () => []);
            const voiceAssignOrder = scoreSegments.slice().sort((a, b) => {
                const aStart = Math.min(...a.points.map(p => p.x));
                const bStart = Math.min(...b.points.map(p => p.x));
                if (a.staff !== b.staff) return a.staff - b.staff;
                if (aStart !== bStart) return aStart - bStart;
                const aEnd = Math.max(...a.points.map(p => p.x));
                const bEnd = Math.max(...b.points.map(p => p.x));
                return aEnd - bEnd;
            });

            for (const seg of voiceAssignOrder) {
                const startX = Math.min(...seg.points.map(p => p.x));
                const endExclusive = Math.max(...seg.points.map(p => p.x)) + 1;
                const voiceEnds = voiceEndsByStaff[seg.staff];

                let chosenVoice = -1;
                let latestReusableEnd = -Infinity;
                for (let v = 0; v < voiceEnds.length; v++) {

                    if (voiceEnds[v] <= startX && voiceEnds[v] > latestReusableEnd) {
                        chosenVoice = v;
                        latestReusableEnd = voiceEnds[v];
                    }
                }

                if (chosenVoice === -1) {
                    chosenVoice = voiceEnds.length;
                    voiceEnds.push(endExclusive);
                } else {
                    voiceEnds[chosenVoice] = endExclusive;
                }

                seg.voice = chosenVoice;
                seg.startX = startX;
                seg.endExclusive = endExclusive;
            }

            const totalVoices = voiceEndsByStaff.reduce((sum, voices) => sum + voices.length, 0);

            const noteEvents = [];
            const staveLines = [];
            const tieLines = [];

            function durationSpecFromBeats(beats) {
                if (beats >= 4) return { duration: 'w', dots: 0, beats: 4 };
                if (beats === 3) return { duration: 'h', dots: 1, beats: 3 };
                if (beats === 2) return { duration: 'h', dots: 0, beats: 2 };
                return { duration: 'q', dots: 0, beats: 1 };
            }

            function splitBeatsToDurationSpecs(totalBeats) {
                const specs = [];
                let remaining = totalBeats;
                while (remaining > 0) {
                    const spec = durationSpecFromBeats(remaining);
                    specs.push(spec);
                    remaining -= spec.beats;
                }
                return specs;
            }

            function quantizePitch(p) {
                return clampInt(Math.round(p), 0, PITCHES_PER_STAFF - 1);
            }

            function interpolatePitch(aPitch, bPitch, aX, bX, x) {
                if (bX <= aX) return quantizePitch(aPitch);
                const t = (x - aX) / (bX - aX);
                return quantizePitch(aPitch + (bPitch - aPitch) * t);
            }

            let eventId = 0;

            for (const seg of scoreSegments) {
                if (!seg.points || seg.points.length < 1) continue;
                const staff = seg.staff;
                const voice = seg.voice ?? 0;

                const ordered = seg.points
                    .map(p => ({ x: p.x, pitch: Math.round(p.pitch) }))
                    .filter(p => p.x >= 0 && p.x < SCORE_GRID_X && p.pitch >= 0 && p.pitch < PITCHES_PER_STAFF)
                    .sort((a, b) => a.x - b.x || a.pitch - b.pitch);

                const points = [];
                for (const p of ordered) {
                    const last = points[points.length - 1];
                    if (last && last.x === p.x && last.pitch === p.pitch) continue;
                    points.push({ ...p });
                }
                if (points.length === 0) continue;

                const runs = [];
                let runStartX = points[0].x;
                let runEndX = points[0].x;
                let runPitch = points[0].pitch;
                for (let i = 1; i < points.length; i++) {
                    const p = points[i];
                    if (p.pitch === runPitch && p.x <= runEndX + 1) {
                        runEndX = p.x;
                    } else {
                        runs.push({ x0: runStartX, x1: runEndX, pitch: runPitch });
                        runStartX = p.x;
                        runEndX = p.x;
                        runPitch = p.pitch;
                    }
                }
                runs.push({ x0: runStartX, x1: runEndX, pitch: runPitch });

                const anchors = [];
                for (const run of runs) {
                    anchors.push({ x: run.x0, pitch: run.pitch, sampledEndX: run.x1, originalX: run.x0 });
                    if (run.x1 > run.x0) {
                        anchors.push({ x: run.x1, pitch: run.pitch, sampledEndX: run.x1, originalX: run.x1 });
                    }
                }
                anchors.sort((a, b) => a.x - b.x || a.pitch - b.pitch);

                const normalizedAnchors = [];
                for (const a of anchors) {
                    const last = normalizedAnchors[normalizedAnchors.length - 1];
                    if (last && last.x === a.x && last.pitch === a.pitch) {
                        last.sampledEndX = Math.max(last.sampledEndX, a.sampledEndX);
                    } else {
                        normalizedAnchors.push({ ...a });
                    }
                }

                const scheduledAnchors = [];
                for (const a of normalizedAnchors) {
                    const prev = scheduledAnchors[scheduledAnchors.length - 1];
                    let scheduledX = a.x;
                    let approxVerticalFromPrev = false;
                    if (prev) {
                        const currentOriginalX = a.originalX ?? a.x;
                        const prevOriginalX = prev.originalX ?? prev.x;
                        const sameOriginalXAsPrev = currentOriginalX === prevOriginalX;

                        if (sameOriginalXAsPrev) {
                            scheduledX = prev.x + 1;
                            approxVerticalFromPrev = true;
                        } else if (scheduledX <= prev.x) {
                            // 直前の真の垂直線を1拍ぶん展開した結果の時間衝突。
                            // 時刻だけ前へ送るが、この接続自体は垂直扱いしない。
                            scheduledX = prev.x + 1;
                        }
                    }
                    scheduledAnchors.push({
                        ...a,
                        x: scheduledX,
                        approxVerticalFromPrev,
                        originalX: a.originalX ?? a.x,
                    });
                }

                const anchorFirstEventIds = [];
                for (let a = 0; a < scheduledAnchors.length; a++) {
                    const anchor = scheduledAnchors[a];
                    const next = scheduledAnchors[a + 1];
                    const spanEndExclusive = next
                        ? Math.max(anchor.x + 1, next.x)
                        : Math.max(anchor.x + 1, anchor.sampledEndX + 1);
                    let remainingBeats = Math.max(1, spanEndExclusive - anchor.x);
                    let cursorX = anchor.x;
                    let firstId = null;
                    let prevChunkId = null;

                    const anchorHasGlissTarget = !!next;
                    while (remainingBeats > 0) {
                        const measureBeat = cursorX % COLS_PER_MEASURE;
                        const room = COLS_PER_MEASURE - measureBeat;
                        const beatsThisMeasure = Math.min(remainingBeats, room);
                        const specs = splitBeatsToDurationSpecs(beatsThisMeasure);

                        for (const spec of specs) {
                            const isFirstChunk = firstId === null;
                            const renderedPitch = anchorHasGlissTarget && !isFirstChunk
                                ? interpolatePitch(anchor.pitch, next.pitch, anchor.x, next.x, cursorX)
                                : anchor.pitch;
                            const midi = PITCH_TABLE[staff]?.[renderedPitch];
                            if (midi === undefined) break;

                            const ev = {
                                id: eventId++,
                                staff,
                                voice,
                                measure: Math.floor(cursorX / COLS_PER_MEASURE),
                                xStart: cursorX,
                                xEndExclusive: cursorX + spec.beats,
                                pitch: renderedPitch,
                                midi,
                                durationBeats: spec.beats,
                                duration: spec.duration,
                                dots: spec.dots,
                                hideNotehead: anchorHasGlissTarget && !isFirstChunk,
                                isContinuationChunk: anchorHasGlissTarget && !isFirstChunk
                            };
                            noteEvents.push(ev);
                            if (firstId === null) firstId = ev.id;
                            if (prevChunkId !== null && !anchorHasGlissTarget) {
                                tieLines.push({ fromEventId: prevChunkId, toEventId: ev.id });
                            }
                            prevChunkId = ev.id;
                            cursorX += spec.beats;
                            remainingBeats -= spec.beats;
                        }
                    }
                    anchorFirstEventIds.push(firstId);
                }

                for (let a = 0; a + 1 < scheduledAnchors.length; a++) {
                    const fromEventId = anchorFirstEventIds[a];
                    const toEventId = anchorFirstEventIds[a + 1];
                    if (fromEventId !== null && toEventId !== null) {
                        staveLines.push({
                            fromEventId,
                            toEventId,
                            staff,
                            voice,
                            approxVertical: scheduledAnchors[a + 1].approxVerticalFromPrev
                        });
                    }
                }
            }

            const actualScoreColsUsed = Math.max(1, ...noteEvents.map(ev => ev.xEndExclusive || (ev.xStart + 1)), 0);
            const actualMeasureCount = Math.max(1, Math.ceil(actualScoreColsUsed / COLS_PER_MEASURE));

            const renderMeasureCount = Math.max(layout.measureCount, actualMeasureCount);
            const renderLayout = {
                ...layout,
                scoreColsUsed: layout.scoreColsUsed,
                measureCount: renderMeasureCount,
                scoreColsPadded: renderMeasureCount * COLS_PER_MEASURE,
                scoreWidth: renderMeasureCount * MEASURE_WIDTH,
                scoreHeight: TOP_MARGIN + layout.staffCount * STAFF_BAND + BOTTOM_MARGIN
            };

            const stats = {
                segments: segments.length,
                scoreSegments: scoreSegments.length,
                voices: totalVoices,
                noteEvents: noteEvents.length,
                stavelines: staveLines.length,
                measures: renderLayout.measureCount,
                staves: renderLayout.staffCount,
                analysisCols: layout.analysisCols,
                analysisRows: layout.analysisRows
            };

            return { noteEvents, staveLines, tieLines, layout: renderLayout, stats };
        }

window.ImageGlissScoreConvert={buildLayout,convert};
})();
