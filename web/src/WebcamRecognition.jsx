/**
 * WebcamRecognition.jsx
 *
 * Live sign language recognition via webcam.
 * Captures video frames and sends them over WebSocket to the backend,
 * which runs MediaPipe hand detection and matches against gloss dictionary.
 * Displays the recognized gloss sequence in real-time.
 */

import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useImperativeHandle,
  forwardRef,
} from "react";

const WS_URL = "ws://127.0.0.1:8000/ws/live_recognition";
const FRAME_INTERVAL_MS = 120; // ~8 fps — balances latency vs. CPU

export default function WebcamRecognition() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const timerRef = useRef(null);
  const streamRef = useRef(null);

  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | connecting | running | error | stopped
  const [currentGloss, setCurrentGloss] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [top3, setTop3] = useState([]);
  const [glossHistory, setGlossHistory] = useState([]); // growing sentence of recognized glosses
  const [lastAdded, setLastAdded] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [frameCount, setFrameCount] = useState(0);

  // -------------------------------------------------------------------
  // Webcam access
  // -------------------------------------------------------------------
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
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

  // -------------------------------------------------------------------
  // WebSocket lifecycle
  // -------------------------------------------------------------------
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

        // Auto-append to history when confidence is high and it differs from last
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
      if (status === "running") setStatus("stopped");
    };

    return ws;
  }, [status]);

  const disconnectWS = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // -------------------------------------------------------------------
  // Frame capture loop
  // -------------------------------------------------------------------
  const captureAndSend = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.readyState < 2) return; // not ready yet

    const ctx = canvas.getContext("2d");
    canvas.width = 320;
    canvas.height = 240;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const reader = new FileReader();
        reader.onloadend = () => {
          const b64 = reader.result.split(",")[1]; // strip data:image/jpeg;base64,
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

  // -------------------------------------------------------------------
  // Start / Stop
  // -------------------------------------------------------------------
  const handleStart = useCallback(async () => {
    setGlossHistory([]);
    setFrameCount(0);
    setCurrentGloss(null);
    setConfidence(0);
    setTop3([]);
    setErrorMsg("");

    const camOk = await startCamera();
    if (!camOk) return;
    connectWS();
    setIsRunning(true);
  }, [startCamera, connectWS]);

  const handleStop = useCallback(() => {
    clearInterval(timerRef.current);
    disconnectWS();
    stopCamera();
    setIsRunning(false);
    setStatus("stopped");
    setCurrentGloss(null);
  }, [disconnectWS, stopCamera]);

  // Capture loop — starts once status = "running"
  useEffect(() => {
    if (status === "running") {
      timerRef.current = setInterval(captureAndSend, FRAME_INTERVAL_MS);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [status, captureAndSend]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      handleStop();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------
  const confPct = Math.round(confidence * 100);
  const confColor =
    confidence > 0.7 ? "#4ade80" : confidence > 0.4 ? "#facc15" : "#f87171";

  return (
    <div className="webcam-panel">
      <h2>3. Live-Gebärdenerkennung (Webcam)</h2>
      <p className="muted" style={{ marginBottom: "0.75rem" }}>
        Halte eine Gebärde vor die Kamera. Das System gleicht Handpositionen
        mit dem Gloss-Wörterbuch ab (MediaPipe + Kosinus-Ähnlichkeit).
      </p>

      <div className="webcam-controls">
        {!isRunning ? (
          <button className="btn-start" onClick={handleStart}>
            ▶ Kamera starten &amp; Erkennung beginnen
          </button>
        ) : (
          <button className="btn-stop" onClick={handleStop}>
            ■ Stopp
          </button>
        )}
        {glossHistory.length > 0 && (
          <button
            className="btn-clear"
            onClick={() => setGlossHistory([])}
            style={{ marginLeft: "0.5rem" }}
          >
            Verlauf löschen
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
            className={isRunning ? "" : "hidden"}
            style={{ width: "100%", borderRadius: "8px", background: "#111" }}
          />
          {!isRunning && (
            <div className="video-placeholder">
              <span>Kamera inaktiv</span>
            </div>
          )}
          {/* Hidden canvas for frame capture */}
          <canvas ref={canvasRef} style={{ display: "none" }} />
          {isRunning && (
            <div className="video-overlay-badge">
              {status === "connecting" ? "Verbinde…" : `Frame ${frameCount}`}
            </div>
          )}
        </div>

        {/* Recognition results */}
        <div className="recognition-results">
          <div className="current-gloss-card">
            <span className="current-gloss-label">Erkannte Geste</span>
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
                  style={{
                    width: `${confPct}%`,
                    background: confColor,
                  }}
                />
                <span className="confidence-label">{confPct}%</span>
              </div>
            )}
          </div>

          {top3.length > 0 && (
            <div className="top3-list">
              <span className="top3-heading">Top-3 Kandidaten</span>
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
          Erkannte Glossenfolge
          <span className="gloss-history-count">
            {glossHistory.length} Glose{glossHistory.length !== 1 ? "n" : ""}
          </span>
        </div>
        <div className="gloss-history-words">
          {glossHistory.length === 0 ? (
            <span className="muted">Noch keine Gesten erkannt…</span>
          ) : (
            glossHistory.map((g, i) => (
              <span
                key={i}
                className={`gloss-chip ${g === lastAdded && i === glossHistory.length - 1 ? "gloss-chip-new" : ""}`}
              >
                {g}
              </span>
            ))
          )}
        </div>
      </div>

      <p className="muted" style={{ fontSize: "0.78rem", marginTop: "0.5rem" }}>
        Erkennungsrate ~8 fps · Konfidenz-Schwelle 60% für Auto-Append ·
        Kein Training nötig (zero-shot via Skelett-Matching)
      </p>
    </div>
  );
}
