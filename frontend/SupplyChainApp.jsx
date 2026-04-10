import { useState, useRef, useEffect, useCallback } from "react";

const API = "http://localhost:8000";

const SUGGESTIONS = [
  "What is the demand forecast for all materials?",
  "Show stockout risk for all materials",
  "Cluster supplier risk",
  "Optimize inventory policy and strategy",
  "What is safety stock?",
];

const themes = {
  light: {
    bg:         "#f7f6f3",
    surface:    "#ffffff",
    surfaceAlt: "#f0efe9",
    border:     "#e2e0d8",
    borderSoft: "#ece9e1",
    text:       "#1a1916",
    textSub:    "#6b6760",
    textMuted:  "#b0ada5",
    accent:     "#c17f3a",
    accentText: "#ffffff",
    userBubble: "#1a1916",
    userText:   "#f7f6f3",
    aiBubble:   "#ffffff",
    aiText:     "#1a1916",
    rawBg:      "#f0efe9",
    rawText:    "#3d7a5a",
    inputBg:    "#ffffff",
    shadow:     "0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
    shadowMd:   "0 4px 12px rgba(0,0,0,0.08)",
    positive:   "#2e7d4f",
    positiveBg: "#edf7f1",
    danger:     "#b03030",
    dangerBg:   "#fdf0f0",
    chartLine:  "#c17f3a",
    chartGrid:  "#e2e0d8",
  },
  dark: {
    bg:         "#1c1b18",
    surface:    "#242320",
    surfaceAlt: "#2c2b27",
    border:     "#3a3834",
    borderSoft: "#323028",
    text:       "#edeae3",
    textSub:    "#8a877f",
    textMuted:  "#504d47",
    accent:     "#d4904a",
    accentText: "#1c1b18",
    userBubble: "#edeae3",
    userText:   "#1c1b18",
    aiBubble:   "#242320",
    aiText:     "#edeae3",
    rawBg:      "#1a1916",
    rawText:    "#6daa84",
    inputBg:    "#2c2b27",
    shadow:     "0 1px 3px rgba(0,0,0,0.3)",
    shadowMd:   "0 4px 12px rgba(0,0,0,0.4)",
    positive:   "#6daa84",
    positiveBg: "#1a2820",
    danger:     "#d07070",
    dangerBg:   "#2a1818",
    chartLine:  "#d4904a",
    chartGrid:  "#3a3834",
  },
};

// ── Inject global CSS ──────────────────────────────────────────────────────────
function injectBase() {
  if (document.getElementById("sc-base")) return;
  const s = document.createElement("style");
  s.id = "sc-base";
  s.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@300;400;500&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body, #root { height: 100%; }
    body { font-family: 'DM Sans', sans-serif; font-size: 15px; line-height: 1.6; -webkit-font-smoothing: antialiased; }
    textarea { font-family: inherit; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    @keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes dot { 0%, 60%, 100% { transform: translateY(0); opacity: 0.3; } 30% { transform: translateY(-4px); opacity: 1; } }
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  `;
  document.head.appendChild(s);
}

// ── Icons ──────────────────────────────────────────────────────────────────────
const SendIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/>
  </svg>
);
const SunIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="5"/>
    <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
    <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
  </svg>
);
const MoonIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
  </svg>
);
const UploadIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
);
const ChevronIcon = ({ open }) => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
    style={{ transform: open ? "rotate(90deg)" : "none", transition: "transform .2s" }}>
    <polyline points="9 18 15 12 9 6"/>
  </svg>
);
const GridIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
    <rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
  </svg>
);
const BackIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>
  </svg>
);
const SpinnerIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"
    style={{ animation: "spin 0.8s linear infinite" }}>
    <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
  </svg>
);
const DownloadIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
  </svg>
);
const ImageIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>
    <polyline points="21 15 16 10 5 21"/>
  </svg>
);

// ── Graph image with download button ─────────────────────────────────────────
function GraphImage({ base64, t, label = "chart" }) {
  const [expanded, setExpanded] = useState(false);

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = `data:image/png;base64,${base64}`;
    a.download = `supply_chain_${label}_${Date.now()}.png`;
    a.click();
  };

  return (
    <div style={{ marginTop: 12 }}>
      {/* Toolbar */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        marginBottom: 6,
      }}>
        <span style={{ fontSize: "0.7rem", color: t.textMuted, display: "flex", alignItems: "center", gap: 4 }}>
          <ImageIcon /> Analysis Chart
        </span>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            onClick={() => setExpanded(v => !v)}
            style={{
              background: t.surfaceAlt, border: `1px solid ${t.border}`,
              borderRadius: 6, padding: "3px 9px", cursor: "pointer",
              fontSize: "0.7rem", color: t.textSub, fontFamily: "'DM Sans', sans-serif",
              display: "flex", alignItems: "center", gap: 4, transition: "all .15s",
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = t.textMuted; e.currentTarget.style.color = t.text; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = t.border; e.currentTarget.style.color = t.textSub; }}
          >
            {expanded ? "Collapse" : "Expand"}
          </button>
          <button
            onClick={handleDownload}
            style={{
              background: t.accent, border: "none",
              borderRadius: 6, padding: "3px 9px", cursor: "pointer",
              fontSize: "0.7rem", color: t.accentText, fontFamily: "'DM Sans', sans-serif",
              display: "flex", alignItems: "center", gap: 4, transition: "opacity .15s",
            }}
            onMouseEnter={e => { e.currentTarget.style.opacity = "0.85"; }}
            onMouseLeave={e => { e.currentTarget.style.opacity = "1"; }}
          >
            <DownloadIcon /> Download
          </button>
        </div>
      </div>

      {/* Image */}
      <div style={{
        border: `1px solid ${t.border}`, borderRadius: 8, overflow: "hidden",
        background: "#fff", cursor: "pointer",
        maxHeight: expanded ? "none" : 320,
        transition: "max-height .3s ease",
      }}
        onClick={() => setExpanded(v => !v)}
      >
        <img
          src={`data:image/png;base64,${base64}`}
          alt="Analysis chart"
          style={{ width: "100%", display: "block", objectFit: "contain" }}
        />
      </div>
      {!expanded && (
        <div style={{ textAlign: "center", fontSize: "0.68rem", color: t.textMuted, marginTop: 3 }}>
          Click to expand
        </div>
      )}
    </div>
  );
}

// ── Shared button style helper ─────────────────────────────────────────────────
function Btn({ onClick, disabled, children, variant = "default", t, style = {} }) {
  const base = {
    display: "inline-flex", alignItems: "center", gap: 6,
    border: "1px solid", borderRadius: 8, cursor: disabled ? "not-allowed" : "pointer",
    fontSize: "0.8rem", fontFamily: "'DM Sans', sans-serif",
    padding: "8px 14px", transition: "all .15s", opacity: disabled ? 0.5 : 1,
    ...style,
  };
  const variants = {
    default: { background: t.surfaceAlt, borderColor: t.border, color: t.textSub },
    primary: { background: t.accent,     borderColor: t.accent, color: t.accentText },
    danger:  { background: "transparent", borderColor: t.border, color: t.textMuted },
  };
  const [hover, setHover] = useState(false);
  const hoverVariants = {
    default: { borderColor: t.textMuted, color: t.text },
    primary: { background: t.accent, opacity: 0.9 },
    danger:  { borderColor: "#c0392b", color: "#c0392b" },
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{ ...base, ...variants[variant], ...(hover && !disabled ? hoverVariants[variant] : {}) }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      {children}
    </button>
  );
}

// ── TypingDots ─────────────────────────────────────────────────────────────────
function TypingDots({ t }) {
  return (
    <div style={{ display: "flex", gap: 5, alignItems: "center", padding: "4px 2px" }}>
      {[0, 1, 2].map(i => (
        <span key={i} style={{
          width: 6, height: 6, borderRadius: "50%",
          background: t.textMuted, display: "inline-block",
          animation: `dot 1.2s ease-in-out ${i * 0.18}s infinite`,
        }} />
      ))}
    </div>
  );
}

// ── PreviewTable ───────────────────────────────────────────────────────────────
function PreviewTable({ rows, t }) {
  const [open, setOpen] = useState(false);
  if (!rows?.length) return null;
  const cols = Object.keys(rows[0]);
  return (
    <div style={{ marginTop: 10, border: `1px solid ${t.border}`, borderRadius: 8, overflow: "hidden" }}>
      <button onClick={() => setOpen(v => !v)} style={{
        width: "100%", background: t.surfaceAlt, border: "none",
        color: t.textSub, fontSize: "0.78rem", padding: "8px 12px",
        textAlign: "left", cursor: "pointer",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        fontFamily: "'DM Sans', sans-serif",
      }}>
        <span>Preview data</span>
        <ChevronIcon open={open} />
      </button>
      {open && (
        <div style={{ maxHeight: 160, overflowY: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.73rem" }}>
            <thead>
              <tr>{cols.map(c => (
                <th key={c} style={{ background: t.surfaceAlt, color: t.textMuted, padding: "5px 10px", textAlign: "left", whiteSpace: "nowrap", fontWeight: 500, borderBottom: `1px solid ${t.border}` }}>{c}</th>
              ))}</tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i}>{cols.map(c => (
                  <td key={c} style={{ padding: "4px 10px", borderBottom: `1px solid ${t.borderSoft}`, color: t.textSub, whiteSpace: "nowrap" }}>{String(row[c] ?? "")}</td>
                ))}</tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Sidebar ────────────────────────────────────────────────────────────────────
function Sidebar({ t, meta, onUpload, onSuggestion, onClear, loading }) {
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);

  const handleFile = useCallback((file) => {
    if (file?.name?.endsWith(".xlsx")) onUpload(file);
  }, [onUpload]);

  return (
    <aside style={{
      width: 260, flexShrink: 0,
      background: t.surface, borderRight: `1px solid ${t.border}`,
      display: "flex", flexDirection: "column", overflow: "hidden",
    }}>
      <div style={{ padding: "20px 20px 16px", borderBottom: `1px solid ${t.border}` }}>
        <div style={{ fontFamily: "'Lora', serif", fontSize: "1rem", fontWeight: 500, color: t.text }}>Supply Chain</div>
        <div style={{ fontSize: "0.72rem", color: t.textMuted, marginTop: 2, fontWeight: 300 }}>Intelligence platform</div>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "16px" }}>
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
          style={{
            border: `1.5px dashed ${dragging ? t.accent : meta ? t.accent : t.border}`,
            borderRadius: 10, padding: "18px 14px", textAlign: "center",
            cursor: "pointer", transition: "all .2s",
            background: dragging ? `${t.accent}10` : "transparent",
          }}
        >
          <input ref={inputRef} type="file" accept=".xlsx" style={{ display: "none" }}
            onChange={e => handleFile(e.target.files[0])} />
          <div style={{ color: meta ? t.accent : t.textMuted, marginBottom: 8 }}><UploadIcon /></div>
          <div style={{ fontSize: "0.78rem", color: t.textSub, lineHeight: 1.5 }}>
            {meta
              ? <><span style={{ color: t.accent, fontWeight: 500 }}>{meta.filename}</span><br /><span style={{ color: t.textMuted, fontSize: "0.7rem" }}>Click to replace</span></>
              : <>Drop a <strong style={{ color: t.text }}>.xlsx</strong> file here<br /><span style={{ color: t.textMuted, fontSize: "0.7rem" }}>or click to browse</span></>
            }
          </div>
        </div>

        {meta && (
          <div style={{ marginTop: 14 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              {[
                { label: "Rows",      val: meta.rows.toLocaleString() },
                { label: "Columns",   val: meta.columns },
                ...(meta.unique_suppliers != null ? [{ label: "Suppliers", val: meta.unique_suppliers }] : []),
                ...(meta.unique_materials  != null ? [{ label: "Materials",  val: meta.unique_materials  }] : []),
              ].map(({ label, val }) => (
                <div key={label} style={{ background: t.surfaceAlt, border: `1px solid ${t.border}`, borderRadius: 8, padding: "10px 12px" }}>
                  <div style={{ fontSize: "0.68rem", color: t.textMuted, marginBottom: 3 }}>{label}</div>
                  <div style={{ fontSize: "1rem", fontWeight: 500, color: t.text, fontFamily: "'Lora', serif" }}>{val}</div>
                </div>
              ))}
            </div>
            <PreviewTable rows={meta.preview} t={t} />
          </div>
        )}

        <div style={{ height: 1, background: t.border, margin: "18px 0" }} />

        <div style={{ fontSize: "0.7rem", color: t.textMuted, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 10 }}>
          Try asking
        </div>
        {SUGGESTIONS.map(s => (
          <button key={s} onClick={() => onSuggestion(s)} disabled={loading}
            style={{
              display: "block", width: "100%",
              background: "transparent", border: `1px solid ${t.border}`,
              borderRadius: 8, color: t.textSub, fontFamily: "'DM Sans', sans-serif",
              fontSize: "0.8rem", padding: "8px 11px", textAlign: "left",
              cursor: "pointer", marginBottom: 6, lineHeight: 1.4, transition: "all .15s",
            }}
            onMouseEnter={e => { e.currentTarget.style.background = t.surfaceAlt; e.currentTarget.style.color = t.text; e.currentTarget.style.borderColor = t.textMuted; }}
            onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = t.textSub; e.currentTarget.style.borderColor = t.border; }}
          >{s}</button>
        ))}

        <div style={{ height: 1, background: t.border, margin: "16px 0" }} />

        <button onClick={onClear}
          style={{
            width: "100%", background: "transparent",
            border: `1px solid ${t.border}`, borderRadius: 8,
            color: t.textMuted, fontFamily: "'DM Sans', sans-serif",
            fontSize: "0.78rem", padding: "8px", cursor: "pointer", transition: "all .15s",
          }}
          onMouseEnter={e => { e.currentTarget.style.color = "#c0392b"; e.currentTarget.style.borderColor = "#c0392b"; }}
          onMouseLeave={e => { e.currentTarget.style.color = t.textMuted; e.currentTarget.style.borderColor = t.border; }}
        >Clear conversation</button>
      </div>

      <div style={{ padding: "12px 16px", borderTop: `1px solid ${t.border}` }}>
        <div style={{ fontSize: "0.68rem", color: t.textMuted, lineHeight: 1.8 }}>
          Manoj Nayak · Pranoti Patil · Shalakha Bhor<br />MIT WPU
        </div>
      </div>
    </aside>
  );
}

// ── Message bubble ─────────────────────────────────────────────────────────────
function Message({ msg, t }) {
  const isUser = msg.role === "user";

  // Determine graph label from content
  const graphLabel = msg.content?.toLowerCase().includes("demand") ? "demand_forecast"
    : msg.content?.toLowerCase().includes("stockout") ? "stockout_risk"
    : msg.content?.toLowerCase().includes("supplier") ? "supplier_risk"
    : msg.content?.toLowerCase().includes("policy") ? "policy_optimization"
    : "analysis";

  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", margin: "10px 0", animation: "fadeUp .2s ease both" }}>
      {isUser ? (
        <div style={{
          maxWidth: "65%", padding: "11px 15px",
          background: t.userBubble, color: t.userText,
          borderRadius: "16px 4px 16px 16px",
          fontSize: "0.88rem", lineHeight: 1.6, boxShadow: t.shadow,
        }}>{msg.content}</div>
      ) : (
        <div style={{ maxWidth: "80%" }}>
          <div style={{
            padding: "12px 16px",
            background: t.aiBubble, color: t.aiText,
            border: `1px solid ${t.border}`,
            borderRadius: "4px 16px 16px 16px",
            fontSize: "0.88rem", lineHeight: 1.65, boxShadow: t.shadow,
          }}>
            {msg.typing ? <TypingDots t={t} /> : (
              <>
                {/* Raw model output collapsible */}
                {msg.raw_output && msg.raw_output.trim() && (
                  <details style={{ marginBottom: 10 }}>
                    <summary style={{
                      fontSize: "0.7rem", color: t.textMuted, cursor: "pointer",
                      textTransform: "uppercase", letterSpacing: "0.06em", userSelect: "none",
                    }}>
                      Raw model output
                    </summary>
                    <div style={{
                      background: t.rawBg, borderRadius: 6, padding: "10px 12px",
                      marginTop: 6, fontFamily: "monospace", fontSize: "0.73rem",
                      color: t.rawText, whiteSpace: "pre-wrap",
                      maxHeight: 200, overflowY: "auto",
                      border: `1px solid ${t.border}`,
                    }}>
                      {msg.raw_output}
                    </div>
                  </details>
                )}

                {/* Main answer text */}
                <div dangerouslySetInnerHTML={{
                  __html: (msg.content || "")
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                    .replace(/\n/g, "<br>")
                }} />

                {/* Graph with download */}
                {msg.graph && (
                  <GraphImage base64={msg.graph} t={t} label={graphLabel} />
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// DASHBOARD PAGE
// ══════════════════════════════════════════════════════════════════════════════

function KpiCard({ label, value, sub, t }) {
  return (
    <div style={{
      background: t.surface, border: `1px solid ${t.border}`,
      borderRadius: 10, padding: "18px 20px", boxShadow: t.shadow,
    }}>
      <div style={{ fontSize: "0.7rem", color: t.textMuted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>{label}</div>
      <div style={{ fontFamily: "'Lora', serif", fontSize: "1.8rem", fontWeight: 500, color: t.text }}>{value ?? "—"}</div>
      {sub && <div style={{ fontSize: "0.75rem", color: t.textSub, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function SectionHeader({ title, t }) {
  return (
    <div style={{ fontFamily: "'Lora', serif", fontSize: "1rem", fontWeight: 500, color: t.text, marginBottom: 14, marginTop: 28 }}>
      {title}
    </div>
  );
}

function DataTable({ rows, columns, t }) {
  if (!rows?.length) return <div style={{ fontSize: "0.82rem", color: t.textMuted, padding: "16px 0" }}>No data available.</div>;
  const cols = columns || Object.keys(rows[0]);
  return (
    <div style={{ border: `1px solid ${t.border}`, borderRadius: 10, overflow: "hidden", boxShadow: t.shadow }}>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
          <thead>
            <tr>{cols.map(c => (
              <th key={c} style={{ background: t.surfaceAlt, color: t.textMuted, padding: "10px 14px", textAlign: "left", whiteSpace: "nowrap", fontWeight: 500, borderBottom: `1px solid ${t.border}` }}>{c}</th>
            ))}</tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? t.surface : t.surfaceAlt }}>
                {cols.map(c => (
                  <td key={c} style={{ padding: "9px 14px", borderBottom: `1px solid ${t.borderSoft}`, color: t.textSub, whiteSpace: "nowrap" }}>
                    {String(row[c] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AlertBanner({ count, t }) {
  if (!count) return null;
  return (
    <div style={{
      background: t.dangerBg, border: `1px solid ${t.danger}30`,
      borderRadius: 10, padding: "14px 18px", marginBottom: 10,
      display: "flex", alignItems: "center", gap: 12,
    }}>
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: t.danger, flexShrink: 0 }} />
      <div style={{ fontSize: "0.85rem", color: t.danger }}>
        <strong>{count} material{count > 1 ? "s" : ""}</strong> require reorder attention.
      </div>
    </div>
  );
}

function PolicyCompareChart({ naive, smart, label, t }) {
  const max = Math.max(naive, smart, 1);
  const naivePct = (naive / max) * 100;
  const smartPct = (smart / max) * 100;
  const better = smart <= naive;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: "0.72rem", color: t.textMuted, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
        <div style={{ width: 52, fontSize: "0.7rem", color: t.textSub, flexShrink: 0 }}>Naive</div>
        <div style={{ flex: 1, background: t.surfaceAlt, borderRadius: 4, height: 18, overflow: "hidden" }}>
          <div style={{ width: `${naivePct}%`, height: "100%", background: t.danger, borderRadius: 4, transition: "width .6s ease" }} />
        </div>
        <div style={{ width: 80, fontSize: "0.72rem", color: t.textSub, textAlign: "right", flexShrink: 0 }}>
          {typeof naive === "number" ? naive.toLocaleString(undefined, { maximumFractionDigits: 1 }) : naive}
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ width: 52, fontSize: "0.7rem", color: t.textSub, flexShrink: 0 }}>Smart</div>
        <div style={{ flex: 1, background: t.surfaceAlt, borderRadius: 4, height: 18, overflow: "hidden" }}>
          <div style={{ width: `${smartPct}%`, height: "100%", background: better ? t.positive : t.danger, borderRadius: 4, transition: "width .6s ease" }} />
        </div>
        <div style={{ width: 80, fontSize: "0.72rem", color: better ? t.positive : t.danger, textAlign: "right", fontWeight: 500, flexShrink: 0 }}>
          {typeof smart === "number" ? smart.toLocaleString(undefined, { maximumFractionDigits: 1 }) : smart}
        </div>
      </div>
    </div>
  );
}

// ── Dashboard helper: parse JSON or text rows ─────────────────────────────────
function parseRows(rawOutput) {
  if (!rawOutput) return [];
  try {
    const parsed = JSON.parse(rawOutput);
    if (Array.isArray(parsed)) return parsed;
  } catch (_) { /* not JSON */ }
  return rawOutput.split("\n").filter(Boolean).map(line => {
    const obj = {};
    line.split(", ").forEach(part => {
      const [k, ...v] = part.split(": ");
      if (k?.trim()) obj[k.trim()] = v.join(": ").trim();
    });
    return obj;
  }).filter(r => Object.keys(r).length > 1);
}

function Dashboard({ t, sessionId, onBack }) {
  // 4 independent data buckets — one per button
  const [demandData,      setDemandData]      = useState(null);
  const [stockoutData,    setStockoutData]    = useState(null);
  const [supplierData,    setSupplierData]    = useState(null);
  const [policyData,      setPolicyData]      = useState(null);
  const [demandLoading,   setDemandLoading]   = useState(false);
  const [stockoutLoading, setStockoutLoading] = useState(false);
  const [supplierLoading, setSupplierLoading] = useState(false);
  const [policyLoading,   setPolicyLoading]   = useState(false);
  const [demandDone,      setDemandDone]      = useState(false);
  const [stockoutDone,    setStockoutDone]    = useState(false);
  const [supplierDone,    setSupplierDone]    = useState(false);
  const [policyDone,      setPolicyDone]      = useState(false);

  const callAPI = async (question) => {
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question }),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "API error"); }
    return res.json();
  };

  // ── Button 1: LSTM Demand Forecast ───────────────────────────────────────
  const runDemand = async () => {
    setDemandLoading(true);
    try {
      // "demand" keyword → demand branch in backend
      const data = await callAPI("What is the demand forecast for all materials?");
      const rows = parseRows(data.raw_output); // raw_output is now JSON
      setDemandData({ rows, answer: data.answer, graph: data.graph });
      setDemandDone(true);
    } catch (e) {
      setDemandData({ error: e.message });
      setDemandDone(true);
    } finally { setDemandLoading(false); }
  };

  // ── Button 2: Monte Carlo Stockout Risk ──────────────────────────────────
  const runStockout = async () => {
    setStockoutLoading(true);
    try {
      // "stockout" keyword → stockout branch in backend (no "risk" collision issue now)
      const data = await callAPI("Show stockout simulation for all materials");
      const rows = parseRows(data.raw_output); // raw_output is now JSON
      setStockoutData({ rows, answer: data.answer, graph: data.graph });
      setStockoutDone(true);
    } catch (e) {
      setStockoutData({ error: e.message });
      setStockoutDone(true);
    } finally { setStockoutLoading(false); }
  };

  // ── Button 3: Supplier Risk Clustering ───────────────────────────────────
  const runSupplier = async () => {
    setSupplierLoading(true);
    try {
      // "supplier" keyword → supplier branch in backend (placed BEFORE stockout branch)
      const data = await callAPI("Analyze supplier risk and cluster vendors");
      const rows = parseRows(data.raw_output); // raw_output is now JSON
      setSupplierData({ rows, answer: data.answer, graph: data.graph });
      setSupplierDone(true);
    } catch (e) {
      setSupplierData({ error: e.message });
      setSupplierDone(true);
    } finally { setSupplierLoading(false); }
  };

  // ── Button 4: Policy Optimization ────────────────────────────────────────
  const runPolicy = async () => {
    setPolicyLoading(true);
    try {
      const data = await callAPI("Optimize inventory policy and strategy");
      const raw = data.raw_output || "";

      // Grab a single labelled value from the raw text output
      // e.g. "Safety Stock      : 123.45 units"  →  "123.45 units"
      const grabRaw = (label) => {
        const m = raw.match(new RegExp(label + "\\s*:\\s*([^\\n]+)"));
        return m ? m[1].trim() : "—";
      };
      // Grab only the numeric part (strips trailing text like " units" or " days")
      const grabNum = (label) => {
        const raw_val = grabRaw(label);
        const m = raw_val.match(/[\d,.]+/);
        return m ? m[0] : raw_val;
      };

      const allCosts     = [...raw.matchAll(/Total Cost\s*:\s*\$([\d,.]+)/g)];
      const allStockouts = [...raw.matchAll(/Total Stockouts\s*:\s*(\d+)/g)];
      const allAvgInv    = [...raw.matchAll(/Avg Inventory\s*:\s*([\d.]+)/g)];

      setPolicyData({
        raw,
        answer:         data.answer,
        graph:          data.graph,
        recommended:    grabRaw("RECOMMENDED"),
        safetyStock:    grabNum("Safety Stock"),
        rop:            grabNum("Reorder Point"),
        orderQty:       grabNum("Order Quantity"),
        leadTime:       grabNum("Lead Time"),
        serviceLevel:   grabRaw("Service Level"),
        supplierRisk:   grabRaw("Supplier Risk"),
        // Cost Savings line looks like: "$1,234.56 ↓ (saving)"
        costSavings:    grabRaw("Cost Savings"),
        stockoutReduce: grabRaw("Stockout Reduction"),
        naiveCost:      allCosts[0]     ? parseFloat(allCosts[0][1].replace(/,/g, "")) : 0,
        naiveStockouts: allStockouts[0] ? parseInt(allStockouts[0][1]) : 0,
        naiveAvgInv:    allAvgInv[0]    ? parseFloat(allAvgInv[0][1]) : 0,
        smartCost:      allCosts[1]     ? parseFloat(allCosts[1][1].replace(/,/g, "")) : 0,
        smartStockouts: allStockouts[1] ? parseInt(allStockouts[1][1]) : 0,
        smartAvgInv:    allAvgInv[1]    ? parseFloat(allAvgInv[1][1]) : 0,
      });
      setPolicyDone(true);
    } catch (e) {
      setPolicyData({ error: e.message });
      setPolicyDone(true);
    } finally { setPolicyLoading(false); }
  };

  const downloadCSV = (rows, filename) => {
    if (!rows?.length) return;
    const cols = Object.keys(rows[0]);
    const csv = [cols.join(","), ...rows.map(r => cols.map(c => `"${r[c] ?? ""}"`).join(","))].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = filename;
    a.click();
  };

  const demandRows   = demandData?.rows   || [];
  const stockoutRows = stockoutData?.rows || [];
  const supplierRows = supplierData?.rows || [];

  const criticalItems = demandRows.filter(r =>
    (r["Decision"] || r["decision"] || "").toUpperCase().includes("REORDER")
  );
  // Stockout Probability from backend is 0–1 float, so threshold is 0.5 (= 50%)
  const highRiskItems = stockoutRows.filter(r =>
    parseFloat(r["Stockout Probability"] ?? r["stockout_probability"] ?? 0) > 0.5
  );
  const anyDone = demandDone || stockoutDone || supplierDone || policyDone;

  // Helper: render a section's AI summary box
  const AISummary = ({ text }) => text ? (
    <div style={{ marginTop: 14, background: t.surfaceAlt, border: `1px solid ${t.border}`, borderRadius: 10, padding: "14px 16px", fontSize: "0.85rem", color: t.textSub, lineHeight: 1.65 }}>
      <div style={{ fontSize: "0.68rem", color: t.textMuted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>AI Summary</div>
      <div dangerouslySetInnerHTML={{
        __html: text
          .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
          .replace(/\n/g, "<br>")
      }} />
    </div>
  ) : null;

  const ErrorBox = ({ msg }) => msg ? (
    <div style={{ background: t.dangerBg, border: `1px solid ${t.danger}30`, borderRadius: 10, padding: "14px 18px", fontSize: "0.84rem", color: t.danger }}>
      Error: {msg}
    </div>
  ) : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: t.bg, color: t.text, animation: "fadeIn .25s ease" }}>
      {/* Header */}
      <div style={{ background: t.surface, borderBottom: `1px solid ${t.border}`, padding: "14px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <Btn onClick={onBack} t={t} style={{ padding: "6px 12px" }}>
            <BackIcon /> Back to chat
          </Btn>
          <div>
            <span style={{ fontFamily: "'Lora', serif", fontSize: "1.05rem", fontWeight: 500, color: t.text }}>MRP Dashboard</span>
            <span style={{ marginLeft: 10, fontSize: "0.75rem", color: t.textMuted }}>Demand · Stockout · Supplier · Policy</span>
          </div>
        </div>
      </div>

      {/* Body */}
      <div style={{ flex: 1, overflowY: "auto", padding: "28px 32px" }}>
        {!sessionId ? (
          <div style={{ textAlign: "center", padding: "80px 20px", color: t.textMuted }}>
            <div style={{ fontFamily: "'Lora', serif", fontSize: "1.2rem", color: t.textSub, marginBottom: 8 }}>No data loaded</div>
            <div style={{ fontSize: "0.85rem" }}>Go back to the chat and upload an Excel file first.</div>
          </div>
        ) : (
          <>
            {/* ── 4 Action Buttons ─────────────────────────────────────────── */}
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <Btn onClick={runDemand} disabled={demandLoading} variant="primary" t={t}>
                {demandLoading ? <><SpinnerIcon /> Running LSTM...</> : "Run Demand Prediction"}
              </Btn>
              <Btn onClick={runStockout} disabled={stockoutLoading} variant="primary" t={t}>
                {stockoutLoading ? <><SpinnerIcon /> Simulating...</> : "Stockout Risk (Monte Carlo)"}
              </Btn>
              <Btn onClick={runSupplier} disabled={supplierLoading} variant="primary" t={t}>
                {supplierLoading ? <><SpinnerIcon /> Clustering...</> : "Supplier Risk Clustering"}
              </Btn>
              <Btn onClick={runPolicy} disabled={policyLoading} variant="primary" t={t}>
                {policyLoading ? <><SpinnerIcon /> Optimizing...</> : "Run Policy Optimization"}
              </Btn>
            </div>

            {/* ── KPI Strip (shown once any section is done) ───────────────── */}
            {anyDone && (
              <>
                <SectionHeader title="Key Metrics" t={t} />
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 12 }}>
                  {demandDone && (
                    <KpiCard label="Total Materials" value={demandRows.length || "—"} sub="in dataset" t={t} />
                  )}
                  {demandDone && (
                    <KpiCard label="Reorder Alerts" value={criticalItems.length} sub="materials need reorder" t={t} />
                  )}
                  {stockoutDone && (
                    <KpiCard label="High Stockout Risk" value={highRiskItems.length} sub="> 50% probability" t={t} />
                  )}
                  {supplierDone && (
                    <KpiCard
                      label="High-Risk Suppliers"
                      value={supplierRows.filter(r => r["Supplier_Risk"] === "High Risk").length}
                      sub="flagged by clustering"
                      t={t}
                    />
                  )}
                  {policyDone && policyData?.costSavings && policyData.costSavings !== "—" && (
                    <KpiCard label="Policy Savings" value={policyData.costSavings} sub="smart vs naive" t={t} />
                  )}
                </div>
              </>
            )}

            {/* ── Section 1: Demand Forecast ───────────────────────────────── */}
            {demandDone && demandData && (
              <>
                <SectionHeader title="Demand Forecast (LSTM)" t={t} />
                {criticalItems.length > 0 && <AlertBanner count={criticalItems.length} t={t} />}
                {demandData.graph && (
                  <div style={{ marginBottom: 16 }}>
                    <GraphImage base64={demandData.graph} t={t} label="demand_forecast" />
                  </div>
                )}
                <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 10 }}>
                  <Btn onClick={() => downloadCSV(demandRows, "demand_predictions.csv")} t={t}>
                    <DownloadIcon /> Download CSV
                  </Btn>
                </div>
                {demandRows.length > 0
                  ? <DataTable rows={demandRows} t={t} />
                  : <div style={{ fontSize: "0.82rem", color: t.textMuted, padding: "8px 0" }}>No tabular data to display.</div>
                }
                <AISummary text={demandData.answer} />
                <ErrorBox msg={demandData.error} />
              </>
            )}

            {/* ── Section 2: Stockout Risk ─────────────────────────────────── */}
            {stockoutDone && stockoutData && (
              <>
                <SectionHeader title="Stockout Risk Analysis (Monte Carlo)" t={t} />
                {stockoutData.graph && (
                  <div style={{ marginBottom: 16 }}>
                    <GraphImage base64={stockoutData.graph} t={t} label="stockout_risk" />
                  </div>
                )}
                <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 10 }}>
                  <Btn onClick={() => downloadCSV(stockoutRows, "stockout_risk.csv")} t={t}>
                    <DownloadIcon /> Download CSV
                  </Btn>
                </div>
                {stockoutRows.length > 0
                  ? <DataTable rows={stockoutRows} t={t} />
                  : <div style={{ fontSize: "0.82rem", color: t.textMuted, padding: "8px 0" }}>No tabular data to display.</div>
                }
                <AISummary text={stockoutData.answer} />
                <ErrorBox msg={stockoutData.error} />
              </>
            )}

            {/* ── Section 3: Supplier Risk ─────────────────────────────────── */}
            {supplierDone && supplierData && (
              <>
                <SectionHeader title="Supplier Risk Clustering (K-Means)" t={t} />
                {supplierData.graph && (
                  <div style={{ marginBottom: 16 }}>
                    <GraphImage base64={supplierData.graph} t={t} label="supplier_risk" />
                  </div>
                )}
                <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 10 }}>
                  <Btn onClick={() => downloadCSV(supplierRows, "supplier_risk.csv")} t={t}>
                    <DownloadIcon /> Download CSV
                  </Btn>
                </div>
                {supplierRows.length > 0
                  ? <DataTable rows={supplierRows} t={t} />
                  : <div style={{ fontSize: "0.82rem", color: t.textMuted, padding: "8px 0" }}>No tabular data to display.</div>
                }
                <AISummary text={supplierData.answer} />
                <ErrorBox msg={supplierData.error} />
              </>
            )}

            {/* Policy section */}
            {policyDone && policyData && !policyData.error && (
              <>
                <SectionHeader title="Policy Optimization" t={t} />

                <div style={{
                  background: policyData.recommended?.includes("Smart") ? t.positiveBg : t.surfaceAlt,
                  border: `1px solid ${policyData.recommended?.includes("Smart") ? t.positive + "40" : t.border}`,
                  borderRadius: 10, padding: "14px 20px", marginBottom: 16,
                  display: "flex", alignItems: "center", gap: 14,
                }}>
                  <div style={{ width: 10, height: 10, borderRadius: "50%", background: policyData.recommended?.includes("Smart") ? t.positive : t.accent, flexShrink: 0 }} />
                  <div>
                    <div style={{ fontSize: "0.72rem", color: t.textMuted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 2 }}>Recommended Policy</div>
                    <div style={{ fontSize: "1rem", fontWeight: 600, color: t.text, fontFamily: "'Lora', serif" }}>{policyData.recommended || "Smart Policy"}</div>
                  </div>
                </div>

                {policyData.graph && (
                  <div style={{ marginBottom: 16 }}>
                    <GraphImage base64={policyData.graph} t={t} label="policy_optimization" />
                  </div>
                )}

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 12, marginBottom: 20 }}>
                  <KpiCard label="Cost Savings"       value={policyData.costSavings}    sub="vs naive policy"   t={t} />
                  <KpiCard label="Stockout Reduction"  value={policyData.stockoutReduce} sub="fewer events"      t={t} />
                  <KpiCard label="Safety Stock"        value={policyData.safetyStock}    sub="units buffer"      t={t} />
                  <KpiCard label="Reorder Point"       value={policyData.rop}            sub="units trigger"     t={t} />
                  <KpiCard label="Order Quantity"      value={policyData.orderQty}       sub="units per order"   t={t} />
                  <KpiCard label="Service Level"       value={policyData.serviceLevel}   sub="target fill rate"  t={t} />
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
                  <div style={{ background: t.surface, border: `1px solid ${t.border}`, borderRadius: 10, padding: "18px 20px", boxShadow: t.shadow }}>
                    <div style={{ fontSize: "0.8rem", fontWeight: 500, color: t.text, marginBottom: 14 }}>Cost Comparison</div>
                    <PolicyCompareChart naive={policyData.naiveCost}      smart={policyData.smartCost}      label="Total Cost ($)"        t={t} />
                    <PolicyCompareChart naive={policyData.naiveStockouts} smart={policyData.smartStockouts} label="Stockout Events"        t={t} />
                    <PolicyCompareChart naive={policyData.naiveAvgInv}    smart={policyData.smartAvgInv}    label="Avg Inventory (units)"  t={t} />
                  </div>
                  <div style={{ background: t.surface, border: `1px solid ${t.border}`, borderRadius: 10, padding: "18px 20px", boxShadow: t.shadow }}>
                    <div style={{ fontSize: "0.8rem", fontWeight: 500, color: t.text, marginBottom: 14 }}>Policy Parameters</div>
                    {[
                      ["Reorder Point",  policyData.rop],
                      ["Order Quantity", policyData.orderQty],
                      ["Safety Stock",   policyData.safetyStock],
                      ["Lead Time",      policyData.leadTime],
                      ["Service Level",  policyData.serviceLevel],
                      ["Supplier Risk",  policyData.supplierRisk],
                    ].map(([label, val]) => (
                      <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderBottom: `1px solid ${t.borderSoft}` }}>
                        <span style={{ fontSize: "0.78rem", color: t.textMuted }}>{label}</span>
                        <span style={{ fontSize: "0.82rem", fontWeight: 500, color: t.text }}>{val}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <details style={{ marginBottom: 14 }}>
                  <summary style={{ fontSize: "0.72rem", color: t.textMuted, cursor: "pointer", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6, userSelect: "none" }}>
                    View raw simulation output
                  </summary>
                  <div style={{ background: t.rawBg, border: `1px solid ${t.border}`, borderRadius: 8, padding: "14px 16px", fontFamily: "monospace", fontSize: "0.76rem", color: t.rawText, whiteSpace: "pre-wrap", maxHeight: 280, overflowY: "auto", marginTop: 8 }}>
                    {policyData.raw}
                  </div>
                </details>

                {policyData.answer && (
                  <div style={{ background: t.surfaceAlt, border: `1px solid ${t.border}`, borderRadius: 10, padding: "14px 16px", fontSize: "0.85rem", color: t.textSub, lineHeight: 1.65 }}>
                    <div style={{ fontSize: "0.68rem", color: t.textMuted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>AI Recommendation</div>
                    <div dangerouslySetInnerHTML={{
                      __html: (policyData.answer || "")
                        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
                        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                        .replace(/\n/g, "<br>")
                    }} />
                  </div>
                )}
              </>
            )}

            {policyDone && policyData?.error && (
              <div style={{ background: t.dangerBg, border: `1px solid ${t.danger}30`, borderRadius: 10, padding: "14px 18px", marginTop: 10, fontSize: "0.84rem", color: t.danger }}>
                Policy optimization error: {policyData.error}
              </div>
            )}

            {!anyDone && !demandLoading && !stockoutLoading && !supplierLoading && !policyLoading && (
              <div style={{ textAlign: "center", padding: "60px 20px", color: t.textMuted }}>
                <div style={{ fontFamily: "'Lora', serif", fontSize: "1.1rem", color: t.textSub, marginBottom: 8 }}>Ready to analyse</div>
                <div style={{ fontSize: "0.85rem" }}>Click one of the four buttons above to run a model.</div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// ROOT APP
// ══════════════════════════════════════════════════════════════════════════════
export default function App() {
  injectBase();

  const [page, setPage]           = useState("chat");
  const [mode, setMode]           = useState("light");
  const [sessionId, setSessionId] = useState(null);
  const [meta, setMeta]           = useState(null);
  const [messages, setMessages]   = useState([]);
  const [loading, setLoading]     = useState(false);
  const [toast, setToast]         = useState({ text: "", show: false });
  const [input, setInput]         = useState("");
  const bottomRef                 = useRef();
  const textareaRef               = useRef();
  const t = themes[mode];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const showToast = (text, ms = 2600) => {
    setToast({ text, show: true });
    setTimeout(() => setToast(v => ({ ...v, show: false })), ms);
  };

  const handleUpload = async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(`${API}/upload`, { method: "POST", body: fd });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
      const data = await res.json();
      setSessionId(data.session_id);
      setMeta(data);
      showToast(`Loaded ${data.rows.toLocaleString()} rows`);
    } catch (err) {
      showToast("Upload failed: " + err.message);
    }
  };

  const handleSend = async (text) => {
    text = text?.trim();
    if (!text || loading) return;
    if (!sessionId) { showToast("Please upload a file first"); return; }

    setMessages(prev => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", typing: true },
    ]);
    setLoading(true);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "";

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question: text }),
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
      const data = await res.json();

      setMessages(prev => [
        ...prev.slice(0, -1),
        {
          role: "assistant",
          content: data.answer || "No response.",
          raw_output: data.raw_output || null,
          graph: data.graph || null,
          typing: false,
        },
      ]);
    } catch (err) {
      setMessages(prev => [
        ...prev.slice(0, -1),
        { role: "assistant", content: `Something went wrong: ${err.message}`, raw_output: null, graph: null, typing: false },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(input); }
  };

  const handleInputChange = e => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 130) + "px";
  };

  // Dashboard page
  if (page === "dashboard") {
    return (
      <>
        <Dashboard t={t} sessionId={sessionId} onBack={() => setPage("chat")} />
        <div style={{
          position: "fixed", bottom: 24, left: "50%",
          transform: `translateX(-50%) translateY(${toast.show ? 0 : 50}px)`,
          opacity: toast.show ? 1 : 0,
          background: t.surface, border: `1px solid ${t.border}`,
          color: t.textSub, fontSize: "0.8rem",
          padding: "8px 18px", borderRadius: 20,
          transition: "all .25s ease", pointerEvents: "none",
          zIndex: 999, boxShadow: t.shadowMd,
        }}>{toast.text}</div>
      </>
    );
  }

  // Chat page
  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden", background: t.bg, color: t.text }}>
      <Sidebar t={t} meta={meta} onUpload={handleUpload} onSuggestion={handleSend} onClear={() => setMessages([])} loading={loading} />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>
        {/* Header */}
        <div style={{
          padding: "14px 28px", borderBottom: `1px solid ${t.border}`,
          background: t.surface,
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div>
            <span style={{ fontFamily: "'Lora', serif", fontSize: "1.05rem", fontWeight: 500, color: t.text }}>
              Tata technologies x MIT WPU supply chain assistant
            </span>
            {meta && (
              <span style={{ marginLeft: 12, fontSize: "0.75rem", color: t.textMuted, fontWeight: 300 }}>
                — {meta.filename}
              </span>
            )}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <button
              onClick={() => setPage("dashboard")}
              style={{
                background: t.accent, border: "none",
                borderRadius: 8, padding: "7px 14px", cursor: "pointer",
                color: t.accentText, display: "flex", alignItems: "center", gap: 7,
                fontSize: "0.78rem", fontFamily: "'DM Sans', sans-serif",
                fontWeight: 500, transition: "opacity .15s",
              }}
              onMouseEnter={e => { e.currentTarget.style.opacity = "0.88"; }}
              onMouseLeave={e => { e.currentTarget.style.opacity = "1"; }}
            >
              <GridIcon /> Dashboard
            </button>

            <button
              onClick={() => setMode(m => m === "light" ? "dark" : "light")}
              style={{
                background: t.surfaceAlt, border: `1px solid ${t.border}`,
                borderRadius: 8, padding: "6px 12px", cursor: "pointer",
                color: t.textSub, display: "flex", alignItems: "center", gap: 7,
                fontSize: "0.78rem", fontFamily: "'DM Sans', sans-serif", transition: "all .15s",
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = t.textMuted; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = t.border; }}
            >
              {mode === "light" ? <MoonIcon /> : <SunIcon />}
              {mode === "light" ? "Dark" : "Light"}
            </button>
          </div>
        </div>

        {/* Chat */}
        <div style={{ flex: 1, overflowY: "auto", padding: "24px 32px 12px" }}>
          {!messages.length ? (
            <div style={{ textAlign: "center", padding: "80px 20px", color: t.textMuted }}>
              <div style={{ fontFamily: "'Lora', serif", fontSize: "1.3rem", fontWeight: 400, color: t.textSub, marginBottom: 10 }}>
                What would you like to know?
              </div>
              <div style={{ fontSize: "0.85rem", lineHeight: 1.7, maxWidth: 380, margin: "0 auto" }}>
                Upload a supply chain file and ask about demand forecasts,
                stockout risk, or supplier performance.
              </div>
            </div>
          ) : (
            messages.map((m, i) => <Message key={i} msg={m} t={t} />)
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={{ padding: "12px 32px 20px", borderTop: `1px solid ${t.border}`, background: t.surface }}>
          <div style={{ display: "flex", gap: 10, alignItems: "flex-end", maxWidth: 780, margin: "0 auto" }}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKey}
              placeholder="Ask about demand, risk, or suppliers..."
              rows={1}
              style={{
                flex: 1, background: t.inputBg, border: `1px solid ${t.border}`,
                borderRadius: 10, color: t.text,
                fontSize: "0.9rem", padding: "11px 14px",
                resize: "none", minHeight: 46, maxHeight: 130,
                outline: "none", lineHeight: 1.5, transition: "border-color .15s",
                boxShadow: t.shadow,
              }}
              onFocus={e => { e.target.style.borderColor = t.accent; }}
              onBlur={e => { e.target.style.borderColor = t.border; }}
            />
            <button
              onClick={() => handleSend(input)}
              disabled={loading || !input.trim()}
              style={{
                width: 46, height: 46, borderRadius: 10, flexShrink: 0,
                background: loading || !input.trim() ? t.surfaceAlt : t.accent,
                border: "none",
                cursor: loading || !input.trim() ? "not-allowed" : "pointer",
                color: loading || !input.trim() ? t.textMuted : t.accentText,
                display: "flex", alignItems: "center", justifyContent: "center",
                transition: "all .15s", boxShadow: t.shadow,
              }}
            >
              <SendIcon />
            </button>
          </div>
          <div style={{ textAlign: "center", fontSize: "0.68rem", color: t.textMuted, marginTop: 8 }}>
            Enter to send, Shift+Enter for new line
          </div>
        </div>
      </div>

      {/* Toast */}
      <div style={{
        position: "fixed", bottom: 24, left: "50%",
        transform: `translateX(-50%) translateY(${toast.show ? 0 : 50}px)`,
        opacity: toast.show ? 1 : 0,
        background: t.surface, border: `1px solid ${t.border}`,
        color: t.textSub, fontSize: "0.8rem",
        padding: "8px 18px", borderRadius: 20,
        transition: "all .25s ease", pointerEvents: "none",
        zIndex: 999, boxShadow: t.shadowMd,
      }}>{toast.text}</div>
    </div>
  );
}