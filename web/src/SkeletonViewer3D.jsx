import { useEffect, useRef, useState, useCallback } from "react";

// ── Joint layout (134-D flat vector per frame) ────────────────────────────────
// indices 0-24  → BODY_25 joints  (x at i*2, y at i*2+1)
// indices 25-45 → Left-hand joints
// indices 46-66 → Right-hand joints
const BODY_N = 25;
const HAND_N = 21;

// BODY_25 skeleton edges
const BODY_EDGES = [
  [0, 1],                       // nose → neck
  [1, 2], [2, 3], [3, 4],       // neck → r-shoulder → r-elbow → r-wrist
  [1, 5], [5, 6], [6, 7],       // neck → l-shoulder → l-elbow → l-wrist
  [1, 8],                       // neck → mid-hip
  [8, 9],  [9, 10],  [10, 11],  // mid-hip → r-hip → r-knee → r-ankle
  [8, 12], [12, 13], [13, 14],  // mid-hip → l-hip → l-knee → l-ankle
  [0, 15], [15, 17],            // nose → r-eye → r-ear
  [0, 16], [16, 18],            // nose → l-eye → l-ear
  [14, 19], [19, 20], [14, 21], // l-ankle foot
  [11, 22], [22, 23], [11, 24], // r-ankle foot
];

// Full hand finger edges (21 joints, 0=wrist)
const HAND_EDGES = [
  [0, 1],  [1, 2],  [2, 3],  [3, 4],   // thumb
  [0, 5],  [5, 6],  [6, 7],  [7, 8],   // index
  [0, 9],  [9, 10], [10,11], [11,12],  // middle
  [0, 13], [13,14], [14,15], [15,16],  // ring
  [0, 17], [17,18], [18,19], [19,20],  // pinky
  [5, 9],  [9, 13], [13,17],           // palm knuckle arch
];

// ── Helpers ───────────────────────────────────────────────────────────────────
function getJoint(frame, idx) {
  const x = frame[idx * 2];
  const y = frame[idx * 2 + 1];
  // Undetected joints are stored as (0, 0)
  if (x === 0 && y === 0) return null;
  // MediaPipe clamps out-of-frame joints to exactly y=1.0 at the bottom boundary
  // instead of marking them undetected — suppress only the hard-clamped cases
  if (y >= 0.999) return null;
  return { x, y };
}

function computeBounds(frames) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const frame of frames) {
    for (let i = 0; i < frame.length / 2; i++) {
      const j = getJoint(frame, i);
      if (!j) continue;
      if (j.x < minX) minX = j.x; if (j.x > maxX) maxX = j.x;
      if (j.y < minY) minY = j.y; if (j.y > maxY) maxY = j.y;
    }
  }
  return isFinite(minX) ? { minX, maxX, minY, maxY } : null;
}

// ── Main draw (called each animation frame) ───────────────────────────────────
function drawSkeleton(canvas, frame, bounds) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  const { minX, maxX, minY, maxY } = bounds;
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  // Keep aspect ratio and center in canvas with padding
  const scale = Math.min(W / rangeX, H / rangeY) * 0.82;
  const offX = (W - rangeX * scale) / 2;
  const offY = (H - rangeY * scale) / 2;

  const px = (x) => offX + (x - minX) * scale;
  const py = (y) => offY + (y - minY) * scale;

  // Background
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, W, H);

  const drawEdges = (edges, jointOffset, color, lineW, dotR) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineW;
    ctx.lineCap = "round";
    for (const [a, b] of edges) {
      const ja = getJoint(frame, a + jointOffset);
      const jb = getJoint(frame, b + jointOffset);
      if (!ja || !jb) continue;
      ctx.beginPath();
      ctx.moveTo(px(ja.x), py(ja.y));
      ctx.lineTo(px(jb.x), py(jb.y));
      ctx.stroke();
    }
    // Dots on top
    ctx.fillStyle = color;
    for (const [a, b] of edges) {
      for (const idx of [a, b]) {
        const j = getJoint(frame, idx + jointOffset);
        if (!j) continue;
        ctx.beginPath();
        ctx.arc(px(j.x), py(j.y), dotR, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };

  // Body (white/light)
  drawEdges(BODY_EDGES, 0, "#e0e0e0", 3, 5);

  // Finger bones — use gradient-like multi-colour per finger for clarity
  const FINGER_COLORS_L = ["#ff6b6b","#ffa94d","#ffd43b","#69db7c","#4dabf7"];
  const FINGER_COLORS_R = ["#ff6b6b","#ffa94d","#ffd43b","#69db7c","#4dabf7"];
  // Each finger: thumb(0-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
  const FINGER_GROUPS = [
    [[0,1],[1,2],[2,3],[3,4]],
    [[0,5],[5,6],[6,7],[7,8]],
    [[0,9],[9,10],[10,11],[11,12]],
    [[0,13],[13,14],[14,15],[15,16]],
    [[0,17],[17,18],[18,19],[19,20]],
  ];
  const PALM_ARCH = [[5,9],[9,13],[13,17]];

  const drawHand = (offset, colors) => {
    // Palm arch first
    ctx.strokeStyle = "#aaa";
    ctx.lineWidth = 2;
    for (const [a, b] of PALM_ARCH) {
      const ja = getJoint(frame, a + offset);
      const jb = getJoint(frame, b + offset);
      if (!ja || !jb) continue;
      ctx.beginPath();
      ctx.moveTo(px(ja.x), py(ja.y));
      ctx.lineTo(px(jb.x), py(jb.y));
      ctx.stroke();
    }
    // Fingers
    for (let f = 0; f < 5; f++) {
      ctx.strokeStyle = colors[f];
      ctx.lineWidth = 2.5;
      ctx.lineCap = "round";
      for (const [a, b] of FINGER_GROUPS[f]) {
        const ja = getJoint(frame, a + offset);
        const jb = getJoint(frame, b + offset);
        if (!ja || !jb) continue;
        ctx.beginPath();
        ctx.moveTo(px(ja.x), py(ja.y));
        ctx.lineTo(px(jb.x), py(jb.y));
        ctx.stroke();
      }
      // Knuckle dots
      ctx.fillStyle = colors[f];
      for (const [a, b] of FINGER_GROUPS[f]) {
        for (const idx of [a, b]) {
          const j = getJoint(frame, idx + offset);
          if (!j) continue;
          ctx.beginPath();
          ctx.arc(px(j.x), py(j.y), 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  };

  // Left hand (offset 25), Right hand (offset 46)
  drawHand(BODY_N, FINGER_COLORS_L);
  drawHand(BODY_N + HAND_N, FINGER_COLORS_R);
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function SkeletonViewer3D({ motion }) {
  const canvasRef  = useRef(null);
  const rafRef     = useRef(null);
  const frameRef   = useRef(0);       // actual frame counter (no re-render lag)
  const playingRef = useRef(true);    // ditto for playing state
  const framesRef  = useRef(null);
  const boundsRef  = useRef(null);

  const [playing,  setPlaying]  = useState(true);
  const [frameIdx, setFrameIdx] = useState(0);

  useEffect(() => {
    if (!motion?.keypoints?.length || !canvasRef.current) return;

    const frames = motion.keypoints;
    const fps    = motion.fps || 25;
    const bounds = computeBounds(frames);
    if (!bounds) return;

    framesRef.current  = frames;
    boundsRef.current  = bounds;
    frameRef.current   = 0;
    playingRef.current = true;
    setPlaying(true);
    setFrameIdx(0);

    const canvas     = canvasRef.current;
    const msPerFrame = 1000 / fps;
    let last = performance.now();

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    drawSkeleton(canvas, frames[0], bounds);

    const tick = (now) => {
      rafRef.current = requestAnimationFrame(tick);
      if (playingRef.current && now - last >= msPerFrame) {
        last = now;
        frameRef.current = (frameRef.current + 1) % frames.length;
        drawSkeleton(canvas, frames[frameRef.current], bounds);
        setFrameIdx(frameRef.current);
      }
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(rafRef.current);
  }, [motion]);

  const togglePlay = useCallback(() => {
    playingRef.current = !playingRef.current;
    setPlaying(playingRef.current);
  }, []);

  const restart = useCallback(() => {
    frameRef.current   = 0;
    playingRef.current = true;
    setPlaying(true);
    setFrameIdx(0);
    if (canvasRef.current && framesRef.current && boundsRef.current) {
      drawSkeleton(canvasRef.current, framesRef.current[0], boundsRef.current);
    }
  }, []);

  const handleScrub = useCallback((e) => {
    const idx = Number(e.target.value);
    frameRef.current = idx;
    setFrameIdx(idx);
    if (canvasRef.current && framesRef.current && boundsRef.current) {
      drawSkeleton(canvasRef.current, framesRef.current[idx], boundsRef.current);
    }
  }, []);

  const totalFrames = motion?.keypoints?.length ?? 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        style={{
          display: "block",
          width: "100%",
          borderRadius: "12px",
          background: "#0d1117",
          border: "1px solid #1f2937",
        }}
      />
      <div className="player-controls">
        <button className="ctrl-btn" onClick={restart} title="Restart">&#9646;&#9664;</button>
        <button className="ctrl-btn" onClick={togglePlay} title={playing ? "Pause" : "Play"}>
          {playing ? "⏸" : "▶"}
        </button>
        <input
          type="range"
          className="scrubber"
          min={0}
          max={Math.max(0, totalFrames - 1)}
          value={frameIdx}
          onChange={handleScrub}
        />
        <span className="frame-counter">{frameIdx + 1} / {totalFrames}</span>
      </div>
    </div>
  );
}
