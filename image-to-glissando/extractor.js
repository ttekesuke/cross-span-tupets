(() => {
'use strict';

function otsu(gray) {
    const hist = new Array(256).fill(0);
    gray.forEach(v => hist[Math.max(0, Math.min(255, Math.round(v)))]++);
    const total = gray.length;
    let sum = 0;
    for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sumB = 0, wB = 0, maxVar = 0, thresholdValue = 127;
    for (let i = 0; i < 256; i++) {
        wB += hist[i];
        if (wB === 0) continue;
        const wF = total - wB;
        if (wF === 0) break;
        sumB += i * hist[i];
        const mB = sumB / wB;
        const mF = (sum - sumB) / wF;
        const varBetween = wB * wF * (mB - mF) * (mB - mF);
        if (varBetween > maxVar) { maxVar = varBetween; thresholdValue = i; }
    }
    return thresholdValue;
}

function boxBlurGray(src, w, h, radius) {
    radius = Math.max(0, Math.round(radius));
    if (radius === 0) return src;
    const tmp = new Float32Array(src.length);
    const out = new Float32Array(src.length);
    const span = radius * 2 + 1;

    for (let y = 0; y < h; y++) {
        let sum = 0;
        for (let k = -radius; k <= radius; k++) {
            const x = Math.max(0, Math.min(w - 1, k));
            sum += src[y * w + x];
        }
        for (let x = 0; x < w; x++) {
            tmp[y * w + x] = sum / span;
            const removeX = Math.max(0, x - radius);
            const addX = Math.min(w - 1, x + radius + 1);
            sum += src[y * w + addX] - src[y * w + removeX];
        }
    }

    for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let k = -radius; k <= radius; k++) {
            const y = Math.max(0, Math.min(h - 1, k));
            sum += tmp[y * w + x];
        }
        for (let y = 0; y < h; y++) {
            out[y * w + x] = sum / span;
            const removeY = Math.max(0, y - radius);
            const addY = Math.min(h - 1, y + radius + 1);
            sum += tmp[addY * w + x] - tmp[removeY * w + x];
        }
    }
    return out;
}

function sobelNms(gray, w, h) {
    const mag = new Float32Array(w * h);
    const dir = new Uint8Array(w * h);
    const scale = 255 / 1443;

    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            const i = y * w + x;
            const a00 = gray[i - w - 1], a01 = gray[i - w], a02 = gray[i - w + 1];
            const a10 = gray[i - 1],                         a12 = gray[i + 1];
            const a20 = gray[i + w - 1], a21 = gray[i + w], a22 = gray[i + w + 1];
            const gx = -a00 + a02 - 2 * a10 + 2 * a12 - a20 + a22;
            const gy = -a00 - 2 * a01 - a02 + a20 + 2 * a21 + a22;
            mag[i] = Math.min(255, Math.hypot(gx, gy) * scale);

            let angle = Math.atan2(gy, gx) * 180 / Math.PI;
            if (angle < 0) angle += 180;
            if (angle < 22.5 || angle >= 157.5) dir[i] = 0;
            else if (angle < 67.5) dir[i] = 1;
            else if (angle < 112.5) dir[i] = 2;
            else dir[i] = 3;
        }
    }

    const out = new Float32Array(w * h);
    for (let y = 2; y < h - 2; y++) {
        for (let x = 2; x < w - 2; x++) {
            const i = y * w + x;
            const m = mag[i];
            if (m <= 0) continue;
            let a, b;
            if (dir[i] === 0) { a = mag[i - 1]; b = mag[i + 1]; }
            else if (dir[i] === 1) { a = mag[i - w + 1]; b = mag[i + w - 1]; }
            else if (dir[i] === 2) { a = mag[i - w]; b = mag[i + w]; }
            else { a = mag[i - w - 1]; b = mag[i + w + 1]; }
            if (m >= a && m >= b) out[i] = m;
        }
    }
    return out;
}

function pointSegmentDistance(p, a, b) {
    const dx = b.x - a.x, dy = b.y - a.y;
    if (dx === 0 && dy === 0) return Math.hypot(p.x - a.x, p.y - a.y);
    const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy)));
    return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

function simplifyRdp(points, epsilon) {
    epsilon = Number(epsilon) || 0;
    if (epsilon <= 0 || points.length <= 2) return points.slice();
    const keep = new Uint8Array(points.length);
    keep[0] = 1;
    keep[points.length - 1] = 1;
    const stack = [[0, points.length - 1]];
    while (stack.length) {
        const [start, end] = stack.pop();
        let maxDist = -1;
        let index = -1;
        for (let i = start + 1; i < end; i++) {
            const d = pointSegmentDistance(points[i], points[start], points[end]);
            if (d > maxDist) { maxDist = d; index = i; }
        }
        if (index > start && index < end && maxDist > epsilon) {
            keep[index] = 1;
            stack.push([start, index], [index, end]);
        }
    }
    const out = [];
    for (let i = 0; i < points.length; i++) if (keep[i]) out.push(points[i]);
    return out;
}

function traceLength(points) {
    let len = 0;
    for (let i = 1; i < points.length; i++) len += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
    return len;
}

function clearImageBorder(mask, w, h, margin = 2) {
    margin = Math.max(1, Math.min(margin, Math.floor(Math.min(w, h) / 2)));
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < margin; x++) {
            mask[y * w + x] = 0;
            mask[y * w + (w - 1 - x)] = 0;
        }
    }
    for (let y = 0; y < margin; y++) {
        for (let x = 0; x < w; x++) {
            mask[y * w + x] = 0;
            mask[(h - 1 - y) * w + x] = 0;
        }
    }
}

function traceEdgeGraph(mask, w, h, minLength, simplifyToleranceValue) {
    const total = w * h;
    const dirs = [-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1];
    const usedEdges = new Set();
    const degree = new Uint8Array(total);

    function validNeighbor(idx, n) {
        if (n < 0 || n >= total || !mask[n]) return false;
        const x = idx % w, nx = n % w;
        return Math.abs(nx - x) <= 1;
    }
    function neighbors(idx) {
        const out = [];
        for (const d of dirs) {
            const n = idx + d;
            if (validNeighbor(idx, n)) out.push(n);
        }
        return out;
    }
    function edgeKey(a, b) {
        if (a > b) { const t = a; a = b; b = t; }
        return a * total + b;
    }
    for (let i = 0; i < total; i++) if (mask[i]) degree[i] = neighbors(i).length;

    function traceFrom(a, b) {
        const pts = [{ x: a % w, y: Math.floor(a / w) }, { x: b % w, y: Math.floor(b / w) }];
        usedEdges.add(edgeKey(a, b));
        let prev = a, cur = b;
        let guard = 0;
        while (guard++ < total) {
            if (degree[cur] !== 2) break;
            const ns = neighbors(cur);
            const next = ns[0] === prev ? ns[1] : ns[0];
            if (next == null) break;
            const key = edgeKey(cur, next);
            if (usedEdges.has(key)) break;
            usedEdges.add(key);
            prev = cur;
            cur = next;
            pts.push({ x: cur % w, y: Math.floor(cur / w) });
            if (cur === a) break;
        }
        return pts;
    }

    const traces = [];
    for (let i = 0; i < total; i++) {
        if (!mask[i] || degree[i] === 2) continue;
        for (const n of neighbors(i)) {
            const key = edgeKey(i, n);
            if (usedEdges.has(key)) continue;
            const pts = traceFrom(i, n);
            if (traceLength(pts) >= minLength) traces.push(simplifyRdp(pts, simplifyToleranceValue));
        }
    }
    for (let i = 0; i < total; i++) {
        if (!mask[i]) continue;
        for (const n of neighbors(i)) {
            const key = edgeKey(i, n);
            if (usedEdges.has(key)) continue;
            const pts = traceFrom(i, n);
            if (traceLength(pts) >= minLength) traces.push(simplifyRdp(pts, simplifyToleranceValue));
        }
    }
    return traces.filter(t => t.length >= 2);
}

function extractEdgeTraces(gray, w, h, options) {
    const blurred = boxBlurGray(gray, w, h, Number(options.blurRadius ?? 1));
    const strength = sobelNms(blurred, w, h);
    const mask = new Uint8Array(w * h);
    const t = Number(options.edgeThreshold ?? 48);
    for (let i = 0; i < strength.length; i++) mask[i] = strength[i] >= t ? 1 : 0;
    clearImageBorder(mask, w, h, 2);
    const traces = traceEdgeGraph(mask, w, h, Number(options.minTraceLength ?? 12), Number(options.simplifyTolerance ?? 1.5));
    return { traces, mask };
}

function extractSilhouetteTraces(gray, w, h, options) {
    const mask = new Uint8Array(w * h);
    const t = Number(options.threshold ?? otsu(gray));
    for (let i = 0; i < gray.length; i++) mask[i] = gray[i] < t ? 1 : 0;
    clearImageBorder(mask, w, h, 2);

    const MIN_COMPONENT_AREA = 30;
    const MAX_COMPONENTS = 300;
    function isFG(x, y) {
        if (x < 0 || y < 0 || x >= w || y >= h) return false;
        return mask[y * w + x] === 1;
    }
    function findComponents(value, connectivity) {
        const labels = new Int32Array(w * h).fill(-1);
        const stack = new Int32Array(w * h);
        const comps = [];
        const diag = connectivity === 8;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const startIdx = y * w + x;
                if (mask[startIdx] !== value || labels[startIdx] !== -1) continue;
                let sp = 0;
                stack[sp++] = startIdx;
                labels[startIdx] = 1;
                let count = 0, touchesBorder = false;
                while (sp > 0) {
                    const cur = stack[--sp];
                    const cx = cur % w, cy = Math.floor(cur / w);
                    count++;
                    if (cx === 0 || cy === 0 || cx === w - 1 || cy === h - 1) touchesBorder = true;
                    const xLo = cx > 0, xHi = cx < w - 1, yLo = cy > 0, yHi = cy < h - 1;
                    const push = n => { if (mask[n] === value && labels[n] === -1) { labels[n] = 1; stack[sp++] = n; } };
                    if (xLo) push(cur - 1);
                    if (xHi) push(cur + 1);
                    if (yLo) push(cur - w);
                    if (yHi) push(cur + w);
                    if (diag) {
                        if (xLo && yLo) push(cur - w - 1);
                        if (xHi && yLo) push(cur - w + 1);
                        if (xLo && yHi) push(cur + w - 1);
                        if (xHi && yHi) push(cur + w + 1);
                    }
                }
                comps.push({ startX: x, startY: y, count, touchesBorder });
            }
        }
        return comps;
    }

    const dirs8 = [
        { dx: -1, dy: 0 }, { dx: -1, dy: -1 }, { dx: 0, dy: -1 }, { dx: 1, dy: -1 },
        { dx: 1, dy: 0 }, { dx: 1, dy: 1 }, { dx: 0, dy: 1 }, { dx: -1, dy: 1 }
    ];
    function dirIndexOf(dx, dy) {
        for (let i = 0; i < 8; i++) if (dirs8[i].dx === dx && dirs8[i].dy === dy) return i;
        return 0;
    }
    function traceBoundary(sx, sy, backDx, backDy, maxSteps) {
        const points = [{ x: sx, y: sy }];
        let cx = sx, cy = sy, bIdx = dirIndexOf(backDx, backDy);
        const seen = new Set();
        let steps = 0;
        while (steps++ < maxSteps) {
            const stateKey = cx * 8 + bIdx + cy * w * 8;
            if (seen.has(stateKey)) break;
            seen.add(stateKey);
            let found = -1;
            for (let k = 1; k <= 8; k++) {
                const idx = (bIdx + k) % 8;
                const nx = cx + dirs8[idx].dx, ny = cy + dirs8[idx].dy;
                if (isFG(nx, ny)) { found = idx; break; }
            }
            if (found === -1) break;
            const prevIdx = (found + 7) % 8;
            const prevX = cx + dirs8[prevIdx].dx;
            const prevY = cy + dirs8[prevIdx].dy;
            cx += dirs8[found].dx;
            cy += dirs8[found].dy;
            bIdx = dirIndexOf(prevX - cx, prevY - cy);
            points.push({ x: cx, y: cy });
        }
        return points;
    }

    let fgComps = findComponents(1, 8).filter(c => c.count >= MIN_COMPONENT_AREA);
    let bgComps = findComponents(0, 4).filter(c => !c.touchesBorder && c.count >= MIN_COMPONENT_AREA);
    fgComps.sort((a, b) => b.count - a.count);
    bgComps.sort((a, b) => b.count - a.count);
    fgComps = fgComps.slice(0, MAX_COMPONENTS);
    bgComps = bgComps.slice(0, MAX_COMPONENTS);

    const traces = [];
    for (const c of fgComps) {
        const pts = traceBoundary(c.startX, c.startY, -1, 0, Math.max(2000, c.count * 8));
        if (pts.length >= 2) traces.push(pts);
    }
    for (const c of bgComps) {
        const fx = c.startX - 1, fy = c.startY;
        if (!isFG(fx, fy)) continue;
        const pts = traceBoundary(fx, fy, 1, 0, Math.max(2000, c.count * 8));
        if (pts.length >= 2) traces.push(pts);
    }
    return { traces, mask };
}

function extract(gray, w, h, options = {}) {
    return options.mode === 'silhouette'
        ? extractSilhouetteTraces(gray, w, h, options)
        : extractEdgeTraces(gray, w, h, options);
}

window.ImageGlissExtractor = { otsu, extract };
})();
