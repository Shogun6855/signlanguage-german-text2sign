import React, { useState } from "react";
import SkeletonViewer3D from "./SkeletonViewer3D";
import WebcamRecognition from "./WebcamRecognition";
import NLPAnalysisPanel from "./NLPAnalysisPanel";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [text, setText] = useState("Wie mein Leben aussieht?");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [gloss, setGloss] = useState([]);
  const [segments, setSegments] = useState([]);
  const [motion, setMotion] = useState(null);
  const [motionMode, setMotionMode] = useState(""); // "gloss" | "segment"
  const [activeTab, setActiveTab] = useState("translate"); // "translate" | "webcam" | "nlp"

  const handleTranslate = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setGloss([]);
    setSegments([]);
    setMotion(null);
    setMotionMode("");
    try {
      const res = await fetch(`${API_BASE}/api/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, chained: true }),
      });
      if (!res.ok) throw new Error(`Translate failed: ${res.status}`);
      const data = await res.json();
      setGloss(data.gloss || []);
      setSegments(data.segments || []);

      if (data.gloss && data.gloss.length > 0) {
        // Try gloss-level animation first (true word-level, shorter clips)
        try {
          const gRes = await fetch(`${API_BASE}/api/motion/by_glosses`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ glosses: data.gloss }),
          });
          if (gRes.ok) {
            const gData = await gRes.json();
            if (gData.keypoints && gData.keypoints.length > 0) {
              setMotion(gData);
              setMotionMode("gloss");
              return;
            }
          }
        } catch (_) { /* fall through */ }
      }

      // Fallback: segment-level chained animation
      if (data.segments && data.segments.length > 0) {
        const mRes = await fetch(`${API_BASE}/api/motion/chained`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ segment_ids: data.segments }),
        });
        if (!mRes.ok) throw new Error(`Motion fetch failed: ${mRes.status}`);
        const mData = await mRes.json();
        setMotion(mData);
        setMotionMode("segment");
      }
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="header">
        <h1>DGS Text &rarr; Sign Demo</h1>
        <p>German sentence to German Sign Language, using DGS-Korpus (1a1).</p>
        <nav className="main-tabs">
          <button
            className={`main-tab ${activeTab === "translate" ? "active" : ""}`}
            onClick={() => setActiveTab("translate")}
          >
            ✍️ Text &rarr; Sign
          </button>
          <button
            className={`main-tab ${activeTab === "webcam" ? "active" : ""}`}
            onClick={() => setActiveTab("webcam")}
          >
            📷 Live Recognition
          </button>
          <button
            className={`main-tab ${activeTab === "nlp" ? "active" : ""}`}
            onClick={() => setActiveTab("nlp")}
          >
            🔬 NLP Pipeline
          </button>
        </nav>
      </header>

      <main className="layout">
        {/* ── Tab 1: Text-to-Sign ── */}
        {activeTab === "translate" && (
          <>
            <section className="panel input-panel">
              <h2>1. Input Text</h2>
              <form onSubmit={handleTranslate}>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={3}
                  placeholder="Enter German sentence..."
                />
                <button type="submit" disabled={loading || !text.trim()}>
                  {loading ? "Translating..." : "Translate to Sign Language"}
                </button>
              </form>
              {error && <p className="error">Error: {error}</p>}
              <div className="panel-section">
                <h3>Detected Segments &amp; Glosses</h3>
                {segments.length === 0 ? (
                  <p className="muted">No mapping found yet.</p>
                ) : (
                  <>
                    <p>
                      <strong>Segment{segments.length > 1 ? "s" : " ID"}:</strong>{" "}
                      {segments.join(", ")}
                      {segments.length > 1 && (
                        <span className="chain-badge">{segments.length} chained</span>
                      )}
                    </p>
                    <p>
                      <strong>Gloss sequence:</strong>{" "}
                      {gloss.length ? gloss.join(" · ") : "—"}
                    </p>
                  </>
                )}
              </div>
              {motion && (
                <>
                  <p className="muted" style={{ marginTop: "0.5rem" }}>
                    Frames: {motion.keypoints.length} &middot; FPS: {motion.fps}
                    {motionMode === "gloss" && (
                      <span className="mode-badge mode-gloss">Gloss-level</span>
                    )}
                    {motionMode === "segment" && (
                      <span className="mode-badge mode-segment">Segment-level</span>
                    )}
                  </p>
                  {motion.missing_glosses?.length > 0 && (
                    <p className="missing-warning">
                      ⚠️ No clip for:{" "}
                      <span className="missing-glosses-list">
                        {motion.missing_glosses.join(", ")}
                      </span>
                    </p>
                  )}
                </>
              )}
            </section>

            <section className="panel animation-panel">
              <h2>2. Sign Animation (Skeleton)</h2>
              {motion ? (
                <>
                  <SkeletonViewer3D
                    motion={motion}
                    glossLabels={motion?.gloss_labels ?? []}
                    frameBoundaries={motion?.frame_boundaries ?? []}
                  />
                  <p className="muted" style={{ marginTop: "0.5rem" }}>
                    Body (white) · Fingers colored by type
                  </p>
                </>
              ) : (
                <div className="placeholder-box">
                  <p className="muted">
                    After translation, the sign movement will be displayed here as an interactive 3D skeleton.
                    <br />
                    <span style={{ fontSize: "0.85em" }}>
                      Body (white) · Fingers colored by type
                    </span>
                  </p>
                </div>
              )}
            </section>
          </>
        )}

        {/* ── Tab 2: Webcam Live Recognition ── */}
        {activeTab === "webcam" && (
          <section className="panel webcam-full-panel">
            <WebcamRecognition />
          </section>
        )}

        {/* ── Tab 3: NLP Analysis ── */}
        {activeTab === "nlp" && (
          <section className="panel nlp-full-panel">
            <NLPAnalysisPanel initialText={text} />
          </section>
        )}
      </main>
    </div>
  );
}

export default App;

