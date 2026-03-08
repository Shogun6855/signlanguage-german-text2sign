/**
 * NLPAnalysisPanel.jsx
 *
 * Displays the NLP pipeline analysis for a German text:
 *  - Tokenization stages (word, char, BPE subword, stemmed, lemmatized)
 *  - HMM POS tagging with Viterbi path
 *  - Feature engineering stats (BoW, TF-IDF, PMI)
 *  - N-gram language model scores and bigram perplexity
 *
 * This panel maps to the course exercise requirements:
 *   Lab 1 — Tokenization & Normalization
 *   Feature Engineering Exercise
 *   Exercise 3 — N-gram Language Models
 *   Exercise 4 — HMM POS Tagging / Viterbi
 */

import React, { useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

const TAG_COLORS = {
  NOUN: "#60a5fa",
  VERB: "#34d399",
  ADJ: "#f59e0b",
  ADV: "#a78bfa",
  PRON: "#fb923c",
  DET: "#94a3b8",
  ADP: "#f472b6",
  CONJ: "#22d3ee",
  NUM: "#e2e8f0",
  PUNCT: "#6b7280",
  X: "#4b5563",
};

function TagChip({ word, tag }) {
  return (
    <span
      className="pos-chip"
      style={{ borderBottom: `3px solid ${TAG_COLORS[tag] || "#888"}` }}
      title={tag}
    >
      {word}
      <sub style={{ color: TAG_COLORS[tag] || "#888", fontSize: "0.65em" }}>{tag}</sub>
    </span>
  );
}

function TokenRow({ label, tokens, color }) {
  if (!tokens || tokens.length === 0) return null;
  return (
    <tr>
      <td className="tok-label">{label}</td>
      <td className="tok-samples">
        {tokens.slice(0, 30).map((t, i) => (
          <span
            key={i}
            className="tok-chip"
            style={{ background: color || "#1e293b" }}
          >
            {t}
          </span>
        ))}
        {tokens.length > 30 && (
          <span className="tok-chip muted">+{tokens.length - 30} mehr</span>
        )}
      </td>
      <td className="tok-count">{tokens.length}</td>
    </tr>
  );
}

function NgramTable({ title, rows }) {
  if (!rows || rows.length === 0) return null;
  return (
    <div style={{ marginBottom: "0.75rem" }}>
      <strong>{title}</strong>
      <table className="ngram-table">
        <thead>
          <tr>
            <th>N-Gram</th>
            <th>Häufigkeit</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 10).map((r, i) => (
            <tr key={i}>
              <td>{r.ngram.join(" ")}</td>
              <td>{r.count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function NLPAnalysisPanel({ initialText }) {
  const [text, setText] = useState(initialText || "Wie mein Leben aussieht?");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("tokenization");

  const handleAnalyze = async () => {
    setLoading(true);
    setError("");
    setData(null);
    try {
      const res = await fetch(`${API_BASE}/api/nlp/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const tok = data?.tokenization;
  const pos = data?.pos_tagging;
  const lm = data?.language_model;
  const fe = data?.feature_engineering;

  return (
    <div className="nlp-panel">
      <h2>4. NLP-Pipeline-Analyse</h2>
      <p className="muted">
        Vollständige Analyse: Tokenisierung (Lab 1) · Feature-Engineering ·
        N-Gram-Sprachmodell (Übung 3) · HMM POS-Tagging mit Viterbi (Übung 4)
      </p>

      <div className="nlp-input-row">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Deutschen Text eingeben…"
          className="nlp-text-input"
        />
        <button
          onClick={handleAnalyze}
          disabled={loading || !text.trim()}
          className="nlp-analyze-btn"
        >
          {loading ? "Analysiere…" : "Analysieren"}
        </button>
      </div>
      {error && <p className="error">Fehler: {error}</p>}

      {data && (
        <>
          {/* Tab bar */}
          <div className="nlp-tabs">
            {[
              ["tokenization", "Tokenisierung (Lab 1)"],
              ["pos", "POS-Tagging / Viterbi (Üb. 4)"],
              ["features", "Feature-Engineering"],
              ["lm", "N-Gram-Sprachmodell (Üb. 3)"],
            ].map(([id, label]) => (
              <button
                key={id}
                className={`nlp-tab ${activeTab === id ? "active" : ""}`}
                onClick={() => setActiveTab(id)}
              >
                {label}
              </button>
            ))}
          </div>

          {/* ─── Tab: Tokenization (Lab 1) ─────────────────────────────── */}
          {activeTab === "tokenization" && tok && (
            <div className="nlp-tab-content">
              <div className="tok-stats-row">
                <span>Wörter: <strong>{tok.stats.word_count}</strong></span>
                <span>Sätze: <strong>{tok.stats.sentence_count}</strong></span>
                <span>Zeichen: <strong>{tok.stats.char_count}</strong></span>
                <span>Subwords (BPE): <strong>{tok.stats.subword_count}</strong></span>
                <span>Unique Wörter: <strong>{tok.stats.unique_words}</strong></span>
              </div>

              <table className="tok-table">
                <thead>
                  <tr>
                    <th>Stufe</th>
                    <th>Token (erste 30)</th>
                    <th>#</th>
                  </tr>
                </thead>
                <tbody>
                  <TokenRow
                    label="Wort-Tokenisierung"
                    tokens={tok.word_tokens}
                    color="#1e3a5f"
                  />
                  <TokenRow
                    label="Satz-Tokenisierung"
                    tokens={tok.sentence_tokens}
                    color="#1e3a2f"
                  />
                  <TokenRow
                    label="Zeichen-Tokenisierung"
                    tokens={tok.char_tokens}
                    color="#2d1b4e"
                  />
                  <TokenRow
                    label="Subword (BPE)"
                    tokens={tok.subword_tokens_bpe}
                    color="#3b2a0a"
                  />
                  <TokenRow
                    label="Nach Stoppwort-Entf."
                    tokens={tok.after_stopword_removal}
                    color="#1c3030"
                  />
                  <TokenRow
                    label="Stemming (Snowball)"
                    tokens={tok.stemmed_tokens}
                    color="#3b1c2a"
                  />
                  <TokenRow
                    label="Lemmatisierung"
                    tokens={tok.lemmatized_tokens}
                    color="#1a2a1a"
                  />
                </tbody>
              </table>

              <div className="comparison-note">
                <h4>Vergleich der Tokenisierungsverfahren</h4>
                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th>Verfahren</th>
                      <th>Token-Anzahl</th>
                      <th>Stärken</th>
                      <th>Schwächen</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Wort</td>
                      <td>{tok.word_tokens.length}</td>
                      <td>Einfach, schnell</td>
                      <td>Unbekannte Wörter</td>
                    </tr>
                    <tr>
                      <td>Zeichen</td>
                      <td>{tok.char_tokens.length}</td>
                      <td>Keine OOV-Probleme</td>
                      <td>Sehr lange Sequenzen</td>
                    </tr>
                    <tr>
                      <td>BPE-Subword</td>
                      <td>{tok.subword_tokens_bpe.length}</td>
                      <td>Balance Wort/Zeichen</td>
                      <td>Weniger interpretierbar</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ─── Tab: POS Tagging / Viterbi (Exercise 4) ────────────────── */}
          {activeTab === "pos" && pos && (
            <div className="nlp-tab-content">
              <h3>HMM POS-Tagging mit Viterbi-Dekodierung</h3>
              <div className="pos-tagged-sentence">
                {pos.tagged_pairs.map(({ word, tag }, i) => (
                  <TagChip key={i} word={word} tag={tag} />
                ))}
              </div>

              <div className="pos-legend">
                {Object.entries(TAG_COLORS).map(([tag, color]) => (
                  <span key={tag} className="pos-legend-item">
                    <span
                      className="pos-legend-dot"
                      style={{ background: color }}
                    />
                    {tag}
                  </span>
                ))}
              </div>

              {pos.viterbi_matrix?.tagged_pairs?.length > 0 && (
                <div style={{ marginTop: "1rem" }}>
                  <h4>Viterbi-Matrix (Log-Wahrscheinlichkeiten)</h4>
                  <div style={{ overflowX: "auto" }}>
                    <table className="viterbi-table">
                      <thead>
                        <tr>
                          <th>Tag ↓ / Token →</th>
                          {pos.viterbi_matrix.tokens.map((t, i) => (
                            <th key={i}>{t}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {pos.viterbi_matrix.tags.map((tag, ti) => (
                          <tr key={ti}>
                            <td
                              className="viterbi-tag-cell"
                              style={{ color: TAG_COLORS[tag] }}
                            >
                              {tag}
                            </td>
                            {pos.viterbi_matrix.matrix.map((col, ci) => {
                              const token = pos.viterbi_matrix.tokens[ci];
                              const val = col[token]?.[tag] ?? -999;
                              const isPredicted =
                                pos.viterbi_matrix.predicted_tags[ci] === tag;
                              return (
                                <td
                                  key={ci}
                                  className={`viterbi-cell ${isPredicted ? "viterbi-best" : ""}`}
                                >
                                  {val.toFixed(1)}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="muted" style={{ fontSize: "0.78rem" }}>
                    Fette/hervorgehobene Zellen = Viterbi-Pfad (beste Tag-Sequenz)
                  </p>
                </div>
              )}

              <div style={{ marginTop: "1rem" }}>
                <h4>Übergangswahrscheinlichkeiten P(tag_i | tag_i-1)</h4>
                <div style={{ overflowX: "auto" }}>
                  <table className="small-table">
                    <thead>
                      <tr>
                        <th>Von →</th>
                        {pos.transition_table[0] &&
                          Object.keys(pos.transition_table[0])
                            .filter((k) => k !== "from_tag")
                            .map((t) => <th key={t}>{t}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {pos.transition_table.map((row, i) => (
                        <tr key={i}>
                          <td
                            style={{
                              fontWeight: 600,
                              color: TAG_COLORS[row.from_tag],
                            }}
                          >
                            {row.from_tag}
                          </td>
                          {Object.entries(row)
                            .filter(([k]) => k !== "from_tag")
                            .map(([k, v]) => (
                              <td key={k} className="prob-cell">
                                {v.toFixed(3)}
                              </td>
                            ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div style={{ marginTop: "1rem" }}>
                <h4>Top Emissionswahrscheinlichkeiten P(wort | tag)</h4>
                <div className="emission-grid">
                  {Object.entries(pos.emission_table || {}).map(
                    ([tag, words]) =>
                      words.length > 0 ? (
                        <div key={tag} className="emission-card">
                          <span
                            className="emission-tag"
                            style={{ color: TAG_COLORS[tag] }}
                          >
                            {tag}
                          </span>
                          {words.map((w) => (
                            <div key={w.word} className="emission-word">
                              {w.word}{" "}
                              <span className="emission-count">({w.count})</span>
                            </div>
                          ))}
                        </div>
                      ) : null
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ─── Tab: Feature Engineering ────────────────────────────────── */}
          {activeTab === "features" && fe && (
            <div className="nlp-tab-content">
              <div className="fe-stats-grid">
                <div className="fe-card">
                  <div className="fe-card-title">Bag-of-Words</div>
                  <div className="fe-stat">Vokabular: <strong>{fe.bow?.vocab_size}</strong></div>
                  <div className="fe-stat">Sparsität: <strong>{fe.bow?.sparsity_pct}%</strong></div>
                  <div className="fe-top-words">
                    {fe.bow?.top_20_words?.slice(0, 10).map((w) => (
                      <span key={w} className="bow-word-chip">{w}</span>
                    ))}
                  </div>
                </div>

                <div className="fe-card">
                  <div className="fe-card-title">N-Gramm-Vokabular</div>
                  <table className="small-table">
                    <thead>
                      <tr>
                        <th>N-Gram</th>
                        <th>Vocab-Größe</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Unigram</td>
                        <td>{fe.ngrams?.unigram_vocab}</td>
                      </tr>
                      <tr>
                        <td>Bigram</td>
                        <td>{fe.ngrams?.bigram_vocab}</td>
                      </tr>
                      <tr>
                        <td>Trigram</td>
                        <td>{fe.ngrams?.trigram_vocab}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div className="fe-card">
                  <div className="fe-card-title">Top TF-IDF Terme</div>
                  {fe.tfidf?.top_10_terms?.map(({ word, score }) => (
                    <div key={word} className="tfidf-row">
                      <span className="tfidf-word">{word}</span>
                      <div className="tfidf-bar-wrap">
                        <div
                          className="tfidf-bar"
                          style={{ width: `${Math.min(score * 200, 100)}%` }}
                        />
                      </div>
                      <span className="tfidf-score">{score}</span>
                    </div>
                  ))}
                </div>

                <div className="fe-card">
                  <div className="fe-card-title">Top PPMI-Wortpaare</div>
                  <table className="small-table">
                    <thead>
                      <tr>
                        <th>Wort 1</th>
                        <th>Wort 2</th>
                        <th>PPMI</th>
                      </tr>
                    </thead>
                    <tbody>
                      {fe.pmi?.top_10_pairs?.slice(0, 8).map(({ pair, ppmi }, i) => (
                        <tr key={i}>
                          <td>{pair[0]}</td>
                          <td>{pair[1]}</td>
                          <td>{ppmi}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="fe-comparison-table-wrap">
                <h4>Vergleich der Feature-Repräsentationen</h4>
                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th>Feature-Typ</th>
                      <th>Dimensionalität</th>
                      <th>Sparsität</th>
                      <th>Semantische Info</th>
                      <th>Sprachspezifisch</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>BoW</td>
                      <td>Hoch ({fe.bow?.vocab_size})</td>
                      <td>{fe.bow?.sparsity_pct}%</td>
                      <td>❌</td>
                      <td>Niedrig</td>
                    </tr>
                    <tr>
                      <td>TF-IDF</td>
                      <td>Hoch ({fe.bow?.vocab_size})</td>
                      <td>Hoch</td>
                      <td>⚠️</td>
                      <td>Mittel</td>
                    </tr>
                    <tr>
                      <td>N-Gramme</td>
                      <td>Sehr hoch ({fe.ngrams?.bigram_vocab} Bi)</td>
                      <td>Sehr hoch</td>
                      <td>⚠️ Lokal</td>
                      <td>Mittel</td>
                    </tr>
                    <tr>
                      <td>PMI/PPMI</td>
                      <td>V²</td>
                      <td>Hoch</td>
                      <td>✅ Ko-Okkurrenz</td>
                      <td>Hoch</td>
                    </tr>
                    <tr>
                      <td>Embeddings (MiniLM)</td>
                      <td>Niedrig (384)</td>
                      <td>Niedrig (dicht)</td>
                      <td>✅✅</td>
                      <td>Hoch (multilingual)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ─── Tab: N-Gram Language Model (Exercise 3) ────────────────── */}
          {activeTab === "lm" && lm && (
            <div className="nlp-tab-content">
              <div className="lm-stats-row">
                <div className="lm-stat-card">
                  <span>Trainierungssequenzen</span>
                  <strong>{lm.trained_on_sequences}</strong>
                </div>
                <div className="lm-stat-card">
                  <span>Bigram Perplexität (Train)</span>
                  <strong>{lm.bigram_perplexity_self ?? "—"}</strong>
                </div>
              </div>
              <p className="muted" style={{ fontSize: "0.82rem", marginBottom: "0.5rem" }}>
                Das N-Gram-Sprachmodell ist über alle Glossen-Sequenzen aus dem
                DGS-Korpus trainiert (Laplace-Smoothing). Niedrige Perplexität
                bedeutet bessere Vorhersage der Zeichenabfolge.
              </p>

              <div className="ngram-tables-row">
                <NgramTable title="Unigram Top-10" rows={lm.unigram_ngram_table} />
                <NgramTable title="Bigram Top-10" rows={lm.bigram_ngram_table} />
              </div>

              <div className="smoothing-note">
                <h4>Laplace (Add-One) Smoothing</h4>
                <p className="muted">
                  P(w_n | w_{"{n-1}"}) = (count(w_{"{n-1}"}, w_n) + 1) /
                  (count(w_{"{n-1}"}) + |V|)
                </p>
                <p className="muted">
                  Smoothing verhindert Zero-Probability für ungesehene N-Gramme.
                  Kandidaten-Übersetzungen werden mit dem Bigram-LM bewertet —
                  die Sequenz mit dem höchsten Log-Wahrscheinlichkeitswert wird
                  als "flüssigste" DGS-Abfolge gewählt.
                </p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
