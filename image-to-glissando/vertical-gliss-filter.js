(() => {
'use strict';

const base = window.ImageGlissScoreConvert;
if (!base || typeof base.convert !== 'function' || typeof base.buildLayout !== 'function') return;

const originalConvert = base.convert.bind(base);

function clampInt(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

function collectTrueVerticalRuns(rawTraces, srcW, srcH) {
    const layout = base.buildLayout(srcW, srcH);
    const cellW = srcW / layout.analysisCols;
    const pitchH = srcH / layout.analysisRows;
    const scoreCols = layout.scoreColsUsed;
    const perScoreCol = layout.analysisColsPerScoreCol;
    const pitchesPerStaff = layout.pitchesPerStaff;
    const xEps = Math.max(1e-7, cellW * 1e-5);
    const runs = [];

    for (const trace of rawTraces || []) {
        if (!trace || trace.length < 2) continue;

        let i = 0;
        while (i < trace.length - 1) {
            const x0 = trace[i].x;
            let j = i + 1;
            while (j < trace.length && Math.abs(trace[j].x - x0) <= xEps) j++;

            if (j - i >= 2) {
                let minY = Infinity;
                let maxY = -Infinity;
                for (let k = i; k < j; k++) {
                    minY = Math.min(minY, trace[k].y);
                    maxY = Math.max(maxY, trace[k].y);
                }

                const qx = clampInt(Math.floor(x0 / cellW), 0, layout.analysisCols - 1);
                const qy0 = clampInt(Math.round(minY / pitchH), 0, layout.analysisRows - 1);
                const qy1 = clampInt(Math.round(maxY / pitchH), 0, layout.analysisRows - 1);

                // Pixel stair-steps in a diagonal contour often contain a one-cell vertical step.
                // Only preserve the special near-vertical notation for a source run that spans
                // at least two analysis pitch cells.
                if (Math.abs(qy1 - qy0) >= 2) {
                    const low = Math.min(qy0, qy1);
                    const high = Math.max(qy0, qy1);
                    const firstStaff = Math.floor(low / pitchesPerStaff);
                    const lastStaff = Math.floor(high / pitchesPerStaff);
                    const scoreX = clampInt(Math.floor(qx / perScoreCol), 0, scoreCols - 1);

                    for (let staff = firstStaff; staff <= lastStaff; staff++) {
                        const staffTop = staff * pitchesPerStaff;
                        const staffBottom = staffTop + pitchesPerStaff - 1;
                        const partLow = Math.max(low, staffTop);
                        const partHigh = Math.min(high, staffBottom);
                        if (partHigh <= partLow) continue;
                        runs.push({
                            staff,
                            scoreX,
                            pitchMin: partLow - staffTop,
                            pitchMax: partHigh - staffTop,
                        });
                    }
                }
            }

            // Reuse the junction point as the start of the next run.
            i = Math.max(i + 1, j - 1);
        }
    }

    return runs;
}

base.convert = function(rawTraces, srcW, srcH, segC) {
    const result = originalConvert(rawTraces, srcW, srcH, segC);
    if (!result || !Array.isArray(result.staveLines) || !Array.isArray(result.noteEvents)) return result;

    const trueVerticalRuns = collectTrueVerticalRuns(rawTraces, srcW, srcH);
    const eventsById = new Map(result.noteEvents.map(ev => [ev.id, ev]));

    for (const line of result.staveLines) {
        if (!line.approxVertical) continue;
        const from = eventsById.get(line.fromEventId);
        const to = eventsById.get(line.toEventId);
        if (!from || !to) {
            line.approxVertical = false;
            continue;
        }

        const pitchMin = Math.min(from.pitch, to.pitch);
        const pitchMax = Math.max(from.pitch, to.pitch);
        const sourceX = from.xStart;

        const backedByTrueVerticalRun = trueVerticalRuns.some(run =>
            run.staff === line.staff &&
            Math.abs(run.scoreX - sourceX) <= 1 &&
            run.pitchMin <= pitchMin &&
            run.pitchMax >= pitchMax
        );

        // If the source geometry does not contain a real vertical run here, fall back to a
        // normal VexFlow StaveLine so it reaches the destination note instead of ending in space.
        if (!backedByTrueVerticalRun) {
            line.approxVertical = false;
            continue;
        }

        // A true vertical glissando has no real elapsed horizontal time. Represent it using
        // standard notation: the source pitch becomes a grace note and the arrival remains
        // a normal note. The renderer keeps the original beat occupied by an invisible
        // GhostNote so the rest of the score timing/layout stays unchanged.
        line.verticalGrace = true;
        from.isGraceSource = true;
        from.graceTargetEventId = to.id;
        if (!Array.isArray(to.graceSourceEventIds)) to.graceSourceEventIds = [];
        if (!to.graceSourceEventIds.includes(from.id)) to.graceSourceEventIds.push(from.id);
    }

    return result;
};
})();
