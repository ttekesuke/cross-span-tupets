(() => {
'use strict';

const base = window.ImageGlissScoreRender;
if (!base || typeof base.render !== 'function') return;

const DISPLAY_W = 1280;
const DISPLAY_H = 720;

function midiToVexKey(midi) {
    const names = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b'];
    const pitchClass = ((Math.round(midi) % 12) + 12) % 12;
    const octave = Math.floor(Math.round(midi) / 12) - 1;
    return `${names[pitchClass]}/${octave}`;
}

function render(scoreData, container, viewport) {
    const { noteEvents, staveLines, tieLines = [], layout } = scoreData;
    if (!container || !viewport) return;
    container.innerHTML = '';

    const canvasWidth = layout.scoreWidth;
    const canvasHeight = layout.scoreHeight;
    const NUM_STAVES = layout.staffCount;
    const COLS_PER_MEASURE = layout.colsPerMeasure;
    const MEASURE_WIDTH = layout.measureWidth;
    const STAVE_Y_START = layout.topMargin;
    const STAVE_SPACING = layout.staffBand;
    const renderer = new Vex.Flow.Renderer(container, Vex.Flow.Renderer.Backends.SVG);
    renderer.resize(canvasWidth, canvasHeight);
    const context = renderer.getContext();

    const staveList = [];
    const numMeasures = layout.measureCount;
    for (let s = 0; s < NUM_STAVES; s++) {
        staveList[s] = [];
        for (let m = 0; m < numMeasures; m++) {
            const x = m * MEASURE_WIDTH;
            const y = STAVE_Y_START + s * STAVE_SPACING;
            const stave = new Vex.Flow.Stave(x, y, MEASURE_WIDTH);
            if (m === 0) stave.addClef('treble').addTimeSignature('4/4');
            else stave.setEndBarType(Vex.Flow.Barline.type.NONE);
            stave.setContext(context).draw();
            staveList[s][m] = stave;
        }
    }

    function durationSpecFromBeats(beats) {
        if (beats >= 4) return { duration: 'wr', dots: 0, beats: 4 };
        if (beats === 3) return { duration: 'hr', dots: 1, beats: 3 };
        if (beats === 2) return { duration: 'hr', dots: 0, beats: 2 };
        return { duration: 'qr', dots: 0, beats: 1 };
    }

    function splitBeatsToRestSpecs(totalBeats) {
        const specs = [];
        let remaining = totalBeats;
        while (remaining > 0) {
            const spec = durationSpecFromBeats(remaining);
            specs.push(spec);
            remaining -= spec.beats;
        }
        return specs;
    }

    function addDots(note, count) {
        for (let i = 0; i < count; i++) {
            Vex.Flow.Dot.buildAndAttach([note], { all: true });
        }
    }

    function makeSpacerTickables(beats) {
        const out = [];
        for (const spec of splitBeatsToRestSpecs(beats)) {
            const rest = new Vex.Flow.StaveNote({ keys: ['b/4'], duration: spec.duration });
            if (spec.dots) addDots(rest, spec.dots);
            out.push(rest);
        }
        return out;
    }

    function makeInvisibleDuration(ev) {
        return new Vex.Flow.GhostNote({ duration: ev.duration });
    }

    function hideNoteheadKeepStem(note) {
        try {
            if (typeof note.setKeyStyle === 'function') {
                note.setKeyStyle(0, { fillStyle: 'rgba(0,0,0,0)', strokeStyle: 'rgba(0,0,0,0)' });
            }
        } catch (e) {}
        const noteHeads = note.noteHeads || note.note_heads || [];
        for (const nh of noteHeads) {
            try {
                if (nh && typeof nh.setStyle === 'function') {
                    nh.setStyle({ fillStyle: 'rgba(0,0,0,0)', strokeStyle: 'rgba(0,0,0,0)' });
                }
            } catch (e) {}
        }
        try {
            if (typeof note.setLedgerLineStyle === 'function') {
                note.setLedgerLineStyle({ strokeStyle: 'rgba(0,0,0,0)' });
            }
        } catch (e) {}
    }

    function noteX(note) {
        return typeof note.getAbsoluteX === 'function' ? note.getAbsoluteX() : (note.absoluteX || 0);
    }

    function noteY(note) {
        if (typeof note.getYs === 'function') {
            const ys = note.getYs();
            return Array.isArray(ys) && ys.length ? ys[0] : 0;
        }
        return Array.isArray(note.ys) && note.ys.length ? note.ys[0] : 0;
    }

    function drawApproxVerticalStaveLine(svg, fromNote, toNote) {
        if (!svg || !fromNote || !toNote) return;
        const ns = 'http://www.w3.org/2000/svg';
        const line = document.createElementNS(ns, 'line');
        const x1 = noteX(fromNote) + 6;
        const y1 = noteY(fromNote);
        const x2 = x1 + 6;
        const y2 = noteY(toNote);
        line.setAttribute('x1', String(x1));
        line.setAttribute('y1', String(y1));
        line.setAttribute('x2', String(x2));
        line.setAttribute('y2', String(y2));
        line.setAttribute('stroke', '#000');
        line.setAttribute('stroke-width', '1.5');
        line.setAttribute('stroke-linecap', 'round');
        svg.appendChild(line);
    }

    function drawGraceVerticalGliss(svg, graceNote, mainNote) {
        if (!svg || !graceNote || !mainNote) return;
        const ns = 'http://www.w3.org/2000/svg';
        const line = document.createElementNS(ns, 'line');
        const gx = noteX(graceNote);
        const mx = noteX(mainNote);
        const x1 = gx + 3;
        const y1 = noteY(graceNote);
        const x2 = Math.max(x1 + 3, mx - 3);
        const y2 = noteY(mainNote);
        line.setAttribute('x1', String(x1));
        line.setAttribute('y1', String(y1));
        line.setAttribute('x2', String(x2));
        line.setAttribute('y2', String(y2));
        line.setAttribute('stroke', '#000');
        line.setAttribute('stroke-width', '1.5');
        line.setAttribute('stroke-linecap', 'round');
        svg.appendChild(line);
    }

    const eventsById = new Map(noteEvents.map(ev => [ev.id, ev]));
    const voiceEvents = {};
    for (const ev of noteEvents) {
        const vKey = `${ev.staff}_${ev.voice}_${ev.measure}`;
        if (!voiceEvents[vKey]) voiceEvents[vKey] = [];
        voiceEvents[vKey].push(ev);
    }

    const noteRefMap = {};
    const measureVoices = {};

    for (const vKey of Object.keys(voiceEvents)) {
        const [staff, voice, measure] = vKey.split('_').map(Number);
        const events = voiceEvents[vKey].slice().sort((a, b) => a.xStart - b.xStart);
        const tickables = [];
        let cursorBeats = 0;

        for (const ev of events) {
            const startBeats = ev.xStart - measure * COLS_PER_MEASURE;
            if (startBeats > cursorBeats) {
                tickables.push(...makeSpacerTickables(startBeats - cursorBeats));
                cursorBeats = startBeats;
            }

            if (ev.isGraceSource) {
                tickables.push(makeInvisibleDuration(ev));
                cursorBeats += ev.durationBeats;
                continue;
            }

            const key = midiToVexKey(ev.midi);
            const note = new Vex.Flow.StaveNote({ keys: [key], duration: ev.duration });
            if (ev.dots) addDots(note, ev.dots);
            if (ev.hideNotehead) hideNoteheadKeepStem(note);

            const graceSourceIds = Array.isArray(ev.graceSourceEventIds) ? ev.graceSourceEventIds : [];
            if (graceSourceIds.length > 0) {
                const graceNotes = [];
                for (const sourceId of graceSourceIds) {
                    const sourceEvent = eventsById.get(sourceId);
                    if (!sourceEvent) continue;
                    const graceKey = midiToVexKey(sourceEvent.midi);
                    const graceNote = new Vex.Flow.GraceNote({
                        keys: [graceKey],
                        duration: '8',
                        slash: false
                    });
                    graceNotes.push(graceNote);
                    noteRefMap[sourceId] = graceNote;
                }
                if (graceNotes.length > 0) {
                    note.addModifier(new Vex.Flow.GraceNoteGroup(graceNotes, false), 0);
                }
            }

            tickables.push(note);
            noteRefMap[ev.id] = note;
            cursorBeats += ev.durationBeats;
        }

        if (cursorBeats < COLS_PER_MEASURE) {
            tickables.push(...makeSpacerTickables(COLS_PER_MEASURE - cursorBeats));
        }

        const voiceObj = new Vex.Flow.Voice({ num_beats: 4, beat_value: 4 })
            .setMode(Vex.Flow.Voice.Mode.SOFT);
        voiceObj.addTickables(tickables);

        const bucketKey = `${staff}_${measure}`;
        if (!measureVoices[bucketKey]) measureVoices[bucketKey] = [];
        measureVoices[bucketKey].push(voiceObj);
    }

    for (let s = 0; s < NUM_STAVES; s++) {
        for (let m = 0; m < numMeasures; m++) {
            const bucketKey = `${s}_${m}`;
            const stave = staveList[s]?.[m];
            if (!stave) continue;
            const voices = measureVoices[bucketKey];
            if (!voices || voices.length === 0) {
                const voiceObj = new Vex.Flow.Voice({ num_beats: 4, beat_value: 4 })
                    .setMode(Vex.Flow.Voice.Mode.SOFT);
                voiceObj.addTickables([
                    new Vex.Flow.StaveNote({ keys: ['b/4'], duration: 'wr' })
                ]);
                new Vex.Flow.Formatter().joinVoices([voiceObj]).format([voiceObj], MEASURE_WIDTH - 8);
                voiceObj.draw(context, stave);
            } else {
                const formatter = new Vex.Flow.Formatter();
                formatter.joinVoices(voices).format(voices, MEASURE_WIDTH - 8);
                for (const voiceObj of voices) voiceObj.draw(context, stave);
            }
        }
    }

    for (const tl of tieLines) {
        const fromNote = noteRefMap[tl.fromEventId];
        const toNote = noteRefMap[tl.toEventId];
        if (!fromNote || !toNote) continue;
        try {
            new Vex.Flow.StaveTie({
                first_note: fromNote,
                last_note: toNote,
                first_indices: [0],
                last_indices: [0]
            }).setContext(context).draw();
        } catch (e) {
            console.warn('StaveTie draw skipped', e);
        }
    }

    const svg = container.querySelector('svg');
    for (const sl of staveLines) {
        const fromNote = noteRefMap[sl.fromEventId];
        const toNote = noteRefMap[sl.toEventId];
        if (!fromNote || !toNote) continue;
        try {
            if (sl.verticalGrace) {
                drawGraceVerticalGliss(svg, fromNote, toNote);
            } else if (sl.approxVertical) {
                drawApproxVerticalStaveLine(svg, fromNote, toNote);
            } else {
                const staveLine = new Vex.Flow.StaveLine({
                    first_note: fromNote,
                    last_note: toNote,
                    first_indices: [0],
                    last_indices: [0]
                });
                staveLine.setContext(context).draw();
            }
        } catch (e) {
            console.warn('StaveLine draw skipped', e, sl);
        }
    }

    const scale = Math.min(DISPLAY_W / canvasWidth, DISPLAY_H / canvasHeight);
    const shownW = canvasWidth * scale;
    const shownH = canvasHeight * scale;
    const offsetX = (DISPLAY_W - shownW) / 2;
    const offsetY = (DISPLAY_H - shownH) / 2;

    viewport.style.width = `${DISPLAY_W}px`;
    viewport.style.height = `${DISPLAY_H}px`;
    viewport.style.overflow = 'hidden';
    viewport.style.position = 'relative';
    container.style.width = `${canvasWidth}px`;
    container.style.height = `${canvasHeight}px`;
    container.style.transformOrigin = 'top left';
    container.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;

    if (svg) {
        svg.setAttribute('viewBox', `0 0 ${canvasWidth} ${canvasHeight}`);
        svg.setAttribute('width', String(canvasWidth));
        svg.setAttribute('height', String(canvasHeight));
        svg.style.display = 'block';
        svg.style.maxWidth = 'none';
    }
}

window.ImageGlissScoreRender = { ...base, render };
})();
