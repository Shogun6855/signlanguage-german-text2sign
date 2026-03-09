/**
 * WebcamRecognition.jsx
 *
 * Live sign language recognition.
 * Two modes:
 *   • Live Camera  — captures webcam frames via getUserMedia
 *   • Video File   — plays a locally-uploaded video and feeds its frames to
 *                    the same WebSocket backend (useful for testing with
 *                    recorded sign-language clips)
 *
 * Frames are sent as base64-JPEG over WebSocket to the backend, which runs
 * MediaPipe hand detection and matches against the gloss dictionary.
 */

import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
} from "react";

const WS_URL = "ws://127.0.0.1:8000/ws/live_recognition";
const FRAME_INTERVAL_MS = 120; // ~8 fps

export default function WebcamRecognition() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const timerRef = useRef(null);
  const streamRef = useRef(null);   // camera MediaStream
  const videoUrlRef = useRef(null); // object URL for uploaded video file

  const [mode, setMode] = useState("camera"); // "camera" | "video"
  const [videoFile, setVideoFile] = useState(null);

  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | connecting | running | error | stopped
  const [currentGloss, setCurrentGloss] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [top3, setTop3] = useState([]);
  const [glossHistory, setGlossHistory] = useState([]);
  const [lastAdded, setLastAdded] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [frameCount, setFrameCount] = useState(0);

  // ── Camera helpers ────────────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.src = "";
        await videoRef.current.play();
      }
      return true;
    } catch (err) {
      setErrorMsg(`Camera error: ${err.message}`);
      setStatus("error");
      return false;
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // ── Video-file helpers ────────────────────────────────────────────────────
  const startVideoFile = useCallback(async () => {
    if (!videoFile) {
      setErrorMsg("Please select a video file first.");
      setStatus("error");
      return false;
    }
    try {
      const url = URL.createObjectURL(videoFile);
      videoUrlRef.current = url;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
        videoRef.current.src = url;
        await videoRef.current.play();
      }
      return true;
    } catch (err) {
      setErrorMsg(`Video error: ${err.message}`);
      setStatus("error");
      return false;
    }
  }, [videoFile]);

  const stopVideoFile = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.src = "";
    }
    if (videoUrlRef.current) {
      URL.revokeObjectURL(videoUrlRef.current);
      videoUrlRef.current = null;
    }
  }, []);

  // ── WebSocket lifecycle ───────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    setStatus("connecting");

    ws.onopen = () => {
      setStatus("running");
      setErrorMsg("");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.error) {
          setErrorMsg(msg.error);
          setStatus("error");
          return;
        }
        const gloss = msg.gloss ?? null;
        const conf = msg.confidence ?? 0;
        setCurrentGloss(gloss);
        setConfidence(conf);
        setTop3(msg.top3 ?? []);

        if (gloss && conf > 0.6) {
          setGlossHistory((prev) => {
            if (prev.length === 0 || prev[prev.length - 1] !== gloss) {
              setLastAdded(gloss);
              return [...prev, gloss];
            }
            return prev;
          });
        }
      } catch (_) {}
    };

    ws.onerror = () => {
      setErrorMsg("WebSocket error — is the backend running?");
      setStatus("error");
    };

    ws.onclose = () => {
      setStatus((s) => (s === "running" ? "stopped" : s));
    };

    return ws;
  }, []);

  const disconnectWS = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // ── Frame capture loop ────────────────────────────────────────────────────
  const captureAndSend = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.readyState < 2) return;

    const ctx = canvas.getContext("2d");
    canvas.width = 320;
    canvas.height = 240;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const reader = new FileReader();
        reader.onloadend = () => {
          const b64 = reader.result.split(",")[1];
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ frame: b64 }));
            setFrameCount((n) => n + 1);
          }
        };
        reader.readAsDataURL(blob);
      },
      "image/jpeg",
      0.6
    );
  }, []);

  // ── Start / Stop ──────────────────────────────────────────────────────────
  const handleStop = useCallback(() => {
    clearInterval(timerRef.current);
    disconnectWS();
    if (mode === "camera") stopCamera();
    else stopVideoFile();
    setIsRunning(false);
    setStatus("stopped");
    setCurrentGloss(null);
  }, [disconnectWS, stopCamera, stopVideoFile, mode]);

  const handleStart = useCallback(async () => {
    setGlossHistory([]);
    setFrameCount(0);
    setCurrentGloss(null);
    setConfidence(0);
    setTop3([]);
    setErrorMsg("");

    const ok =
      mode === "camera" ? await startCamera() : await startVideoFile();
    if (!ok) return;

    connectWS();
    setIsRunning(true);
  }, [mode, startCamera, startVideoFile, connectWS]);

  // Capture loop starts when WebSocket is open
  useEffect(() => {
    if (status === "running") {
      timerRef.current = setInterval(captureAndSend, FRAME_INTERVAL_MS);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [status, captureAndSend]);

  // Auto-stop when video file finishes playing
  const handleVideoEnded = useCallback(() => {
    if (mode === "video" && isRunning) handleStop();
  }, [mode, isRunning, handleStop]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      disconnectWS();
      stopCamera();
      stopVideoFile();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Switch mode → reset everything
  const handleModeSwitch = useCallback(
    (newMode) => {
      if (isRunning) handleStop();
      setMode(newMode);
      setVideoFile(null);
      setErrorMsg("");
      setStatus("idle");
      setGlossHistory([]);
      setCurrentGloss(null);
      setConfidence(0);
      setTop3([]);
      setFrameCount(0);
    },
    [isRunning, handleStop]
  );

  // ── Render ────────────────────────────────────────────────────────────────
  const confPct = Math.round(confidence * 100);
  const confColor =
    confidence > 0.7 ? "#4ade80" : confidence > 0.4 ? "#facc15" : "#f87171";

  return (
    <div className="webcam-panel">
      <h2>3. Live Sign Recognition</h2>
      <p className="muted" style={{ marginBottom: "0.75rem" }}>
        Hold a sign in front of the camera or upload a video file. The
        system matches hand positions against the gloss dictionary (MediaPipe +
        cosine similarity).
      </p>

      {/* Mode toggle */}
      <div className="mode-toggle">
        <button
          className={`mode-btn ${mode === "camera" ? "active" : ""}`}
          onClick={() => handleModeSwitch("camera")}
          disabled={isRunning}
        >
          Live Camera
        </button>
        <button
          className={`mode-btn ${mode === "video" ? "active" : ""}`}
          onClick={() => handleModeSwitch("video")}
          disabled={isRunning}
        >
          Video File
        </button>
      </div>

      {/* File picker — only in video mode, only when not running */}
      {mode === "video" && !isRunning && (
        <div className="file-input-row">
          <label className="file-label">
            {videoFile ? videoFile.name : "Select video file…"}
            <input
              type="file"
              accept="video/*"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0] ?? null;
                setVideoFile(f);
                setErrorMsg("");
                setStatus("idle");
              }}
            />
          </label>
        </div>
      )}

      {/* Start / Stop controls */}
      <div className="webcam-controls">
        {!isRunning ? (
          <button
            className="btn-start"
            onClick={handleStart}
            disabled={mode === "video" && !videoFile}
          >
            {mode === "camera"
              ? "▶ Start Camera & Begin Recognition"
              : "▶ Play Video & Begin Recognition"}
          </button>
        ) : (
          <button className="btn-stop" onClick={handleStop}>
            ■ Stop
          </button>
        )}
        {glossHistory.length > 0 && (
          <button
            className="btn-clear"
            onClick={() => setGlossHistory([])}
            style={{ marginLeft: "0.5rem" }}
          >
            Clear History
          </button>
        )}
      </div>

      {errorMsg && (
        <p className="error" style={{ marginTop: "0.5rem" }}>
          ⚠ {errorMsg}
        </p>
      )}

      <div className="webcam-layout">
        {/* Video preview */}
        <div className="video-container">
          <video
            ref={videoRef}
            muted
            playsInline
            onEnded={handleVideoEnded}
            className={isRunning ? "" : "hidden"}
            style={{ width: "100%", borderRadius: "8px", background: "#111" }}
          />
          {!isRunning && (
            <div className="video-placeholder">
              <span>
                {mode === "camera" ? "Camera inactive" : "Video inactive"}
              </span>
            </div>
          )}
          <canvas ref={canvasRef} style={{ display: "none" }} />
          {isRunning && (
            <div className="video-overlay-badge">
              {status === "connecting" ? "Connecting…" : `Frame ${frameCount}`}
            </div>
          )}
        </div>

        {/* Recognition results */}
        <div className="recognition-results">
          <div className="current-gloss-card">
            <span className="current-gloss-label">Detected Sign</span>
            <span
              className="current-gloss-value"
              style={{ color: currentGloss ? confColor : "#666" }}
            >
              {currentGloss ?? "—"}
            </span>
            {currentGloss && (
              <div className="confidence-bar-wrap">
                <div
                  className="confidence-bar"
                  style={{ width: `${confPct}%`, background: confColor }}
                />
                <span className="confidence-label">{confPct}%</span>
              </div>
            )}
          </div>

          {top3.length > 0 && (
            <div className="top3-list">
              <span className="top3-heading">Top-3 Candidates</span>
              {top3.map(({ gloss, score }, i) => (
                <div key={i} className="top3-row">
                  <span className="top3-rank">#{i + 1}</span>
                  <span className="top3-gloss">{gloss}</span>
                  <span className="top3-score">{Math.round(score * 100)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Recognized gloss sentence */}
      <div className="gloss-history-panel">
        <div className="gloss-history-heading">
          Detected Gloss Sequence
          <span className="gloss-history-count">
            {glossHistory.length} Gloss{glossHistory.length !== 1 ? "es" : ""}
          </span>
        </div>
        <div className="gloss-history-words">
          {glossHistory.length === 0 ? (
            <span className="muted">No signs detected yet…</span>
          ) : (
            glossHistory.map((g, i) => (
              <span
                key={i}
                className={`gloss-chip ${
                  g === lastAdded && i === glossHistory.length - 1
                    ? "gloss-chip-new"
                    : ""
                }`}
              >
                {g}
              </span>
            ))
          )}
        </div>
      </div>

      <p className="muted" style={{ fontSize: "0.78rem", marginTop: "0.5rem" }}>
        Recognition rate ~8 fps · Confidence threshold 60% for auto-append ·
        No training needed (zero-shot via skeleton matching)
        {mode === "video" && " · Video mode: File is analyzed frame-by-frame"}
      </p>
    </div>
  );
}
