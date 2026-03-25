import { useState, useRef, useEffect, useCallback } from "react";

const API = "http://localhost:8000";

const SUGGESTIONS = [
  "What is the demand forecast for all materials?",
  "Show stockout risk for all materials",
  "Cluster supplier risk",
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

// ── Tiny sparkline SVG chart ───────────────────────────────────────────────────
function Sparkline({ values, color, width = 120, height = 36 }) {
  if (!values?.length) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: "visible" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
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
        <div style={{ maxWidth: "75%" }}>
          <div style={{
            padding: "12px 16px",
            background: t.aiBubble, color: t.aiText,
            border: `1px solid ${t.border}`,
            borderRadius: "4px 16px 16px 16px",
            fontSize: "0.88rem", lineHeight: 1.65, boxShadow: t.shadow,
          }}>
            {msg.typing ? <TypingDots t={t} /> : (
              <>
                {msg.raw && (
                  <div style={{
                    background: t.rawBg, borderRadius: 6, padding: "10px 12px",
                    marginBottom: 12, fontFamily: "monospace",
                    fontSize: "0.76rem", color: t.rawText,
                    whiteSpace: "pre-wrap", wordBreak: "break-all",
                    maxHeight: 180, overflowY: "auto", border: `1px solid ${t.border}`,
                  }}>
                    <div style={{ fontSize: "0.65rem", color: t.textMuted, marginBottom: 5, fontFamily: "'DM Sans', sans-serif", textTransform: "uppercase", letterSpacing: "0.06em" }}>Model output</div>
                    {msg.raw}
                  </div>
                )}
                <div dangerouslySetInnerHTML={{
                  __html: msg.content
                    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
                    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                    .replace(/\n/g, "<br>")
                }} />
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

function BarChart({ data, labelKey, valueKey, color, t, height = 160 }) {
  if (!data?.length) return null;
  const max = Math.max(...data.map(d => d[valueKey])) || 1;
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 6, height, padding: "0 4px" }}>
      {data.slice(0, 14).map((d, i) => {
        const pct = (d[valueKey] / max) * 100;
        return (
          <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4, height: "100%" }}>
            <div style={{ flex: 1, display: "flex", alignItems: "flex-end", width: "100%" }}>
              <div style={{
                width: "100%", height: `${pct}%`, minHeight: 2,
                background: color, borderRadius: "3px 3px 0 0",
                transition: "height .4s ease",
              }} title={`${d[labelKey]}: ${d[valueKey]}`} />
            </div>
            <div style={{ fontSize: "0.58rem", color: t.textMuted, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: "100%", textAlign: "center" }}>
              {String(d[labelKey]).slice(0, 6)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Dashboard({ t, sessionId, onBack, mode }) {
  const [demandData,   setDemandData]   = useState(null);
  const [riskData,     setRiskData]     = useState(null);
  const [demandLoading, setDemandLoading] = useState(false);
  const [riskLoading,   setRiskLoading]   = useState(false);
  const [demandDone,   setDemandDone]   = useState(false);
  const [riskDone,     setRiskDone]     = useState(false);

  const runDemand = async () => {
    setDemandLoading(true);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question: "run lstm demand forecast for all materials" }),
      });
      const data = await res.json();
      // Parse raw model output into rows
      const rows = (data.raw_output || "").split("\n").filter(Boolean).map(line => {
        const obj = {};
        line.split(", ").forEach(part => {
          const [k, ...v] = part.split(": ");
          if (k) obj[k.trim()] = v.join(": ").trim();
        });
        return obj;
      }).filter(r => r["Material"]);
      setDemandData({ rows, raw: data.raw_output, answer: data.answer });
      setDemandDone(true);
    } catch (e) {
      setDemandData({ error: e.message });
    } finally {
      setDemandLoading(false);
    }
  };

  const runRisk = async () => {
    setRiskLoading(true);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question: "show stockout risk for all materials monte carlo simulation" }),
      });
      const data = await res.json();
      const rows = (data.raw_output || "").split("\n").filter(Boolean).map(line => {
        const obj = {};
        line.split(", ").forEach(part => {
          const [k, ...v] = part.split(": ");
          if (k) obj[k.trim()] = v.join(": ").trim();
        });
        return obj;
      }).filter(r => r["Material"]);
      setRiskData({ rows, raw: data.raw_output, answer: data.answer });
      setRiskDone(true);
    } catch (e) {
      setRiskData({ error: e.message });
    } finally {
      setRiskLoading(false);
    }
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

  // Derive KPIs
  const demandRows = demandData?.rows || [];
  const riskRows   = riskData?.rows   || [];
  const avgDemand  = demandRows.length ? Math.round(demandRows.reduce((s, r) => s + parseFloat(r["Demand"] || r["Predicted Demand"] || 0), 0) / demandRows.length) : null;
  const maxDemand  = demandRows.length ? Math.round(Math.max(...demandRows.map(r => parseFloat(r["Demand"] || r["Predicted Demand"] || 0)))) : null;
  const criticalItems = demandRows.filter(r => parseFloat(r["Reorder"] || r["Reorder Qty"] || 0) > 0);
  const highRisk   = riskRows.filter(r => parseFloat((r["Stockout"] || r["Stockout Probability"] || "0").replace("%", "")) > 30);

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
            <span style={{ marginLeft: 10, fontSize: "0.75rem", color: t.textMuted }}>Demand prediction &amp; supplier risk</span>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {!sessionId && (
            <span style={{ fontSize: "0.78rem", color: t.textMuted, alignSelf: "center" }}>No file loaded — go back and upload first.</span>
          )}
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
            {/* Action buttons */}
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <Btn onClick={runDemand} disabled={demandLoading || !sessionId} variant="primary" t={t}>
                {demandLoading ? <><SpinnerIcon /> Running LSTM...</> : "Run Demand Prediction"}
              </Btn>
              <Btn onClick={runRisk} disabled={riskLoading || !sessionId} variant="primary" t={t}>
                {riskLoading ? <><SpinnerIcon /> Analyzing...</> : "Analyze Supplier Risk"}
              </Btn>
            </div>

            {/* KPIs */}
            {(demandDone || riskDone) && (
              <>
                <SectionHeader title="Key Metrics" t={t} />
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 12 }}>
                  <KpiCard label="Avg Demand"     value={avgDemand?.toLocaleString() ?? "—"}  sub="predicted units"       t={t} />
                  <KpiCard label="Max Demand"     value={maxDemand?.toLocaleString() ?? "—"}  sub="single material"       t={t} />
                  <KpiCard label="Total Materials" value={demandRows.length || "—"}             sub="in dataset"            t={t} />
                  <KpiCard label="Reorder Alerts"  value={criticalItems.length || "—"}          sub="need reorder"          t={t} />
                  {riskDone && <KpiCard label="High-Risk Items" value={highRisk.length || "0"} sub="stockout &gt; 30%" t={t} />}
                </div>
              </>
            )}

            {/* Demand section */}
            {demandDone && (
              <>
                <SectionHeader title="Demand Predictions" t={t} />
                {criticalItems.length > 0 && <AlertBanner count={criticalItems.length} t={t} />}

                {demandRows.length > 0 && (
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: "0.75rem", color: t.textMuted, marginBottom: 8 }}>Predicted demand by material</div>
                    <div style={{ background: t.surface, border: `1px solid ${t.border}`, borderRadius: 10, padding: "16px", boxShadow: t.shadow }}>
                      <BarChart
                        data={demandRows}
                        labelKey="Material"
                        valueKey="Demand"
                        color={t.chartLine}
                        t={t}
                        height={140}
                      />
                    </div>
                  </div>
                )}

                <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 10 }}>
                  <Btn onClick={() => downloadCSV(demandRows, "demand_predictions.csv")} t={t}>
                    <DownloadIcon /> Download CSV
                  </Btn>
                </div>
                <DataTable rows={demandRows} t={t} />

                {demandData?.answer && (
                  <div style={{ marginTop: 14, background: t.surfaceAlt, border: `1px solid ${t.border}`, borderRadius: 10, padding: "14px 16px", fontSize: "0.85rem", color: t.textSub, lineHeight: 1.65 }}>
                    <div style={{ fontSize: "0.68rem", color: t.textMuted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>AI Summary</div>
                    <div dangerouslySetInnerHTML={{
                      __html: demandData.answer
                        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
                        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                        .replace(/\n/g, "<br>")
                    }} />
                  </div>
                )}
              </>
            )}

            {/* Risk section */}
            {riskDone && (
              <>
                <SectionHeader title="Supplier Risk Analysis" t={t} />

                <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 10 }}>
                  <Btn onClick={() => downloadCSV(riskRows, "supplier_risk.csv")} t={t}>
                    <DownloadIcon /> Download CSV
                  </Btn>
                </div>
                <DataTable rows={riskRows} t={t} />

                {riskData?.answer && (
                  <div style={{ marginTop: 14, background: t.surfaceAlt, border: `1px solid ${t.border}`, borderRadius: 10, padding: "14px 16px", fontSize: "0.85rem", color: t.textSub, lineHeight: 1.65 }}>
                    <div style={{ fontSize: "0.68rem", color: t.textMuted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>AI Summary</div>
                    <div dangerouslySetInnerHTML={{
                      __html: riskData.answer
                        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
                        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                        .replace(/\n/g, "<br>")
                    }} />
                  </div>
                )}
              </>
            )}

            {/* Empty state */}
            {!demandDone && !riskDone && !demandLoading && !riskLoading && (
              <div style={{ textAlign: "center", padding: "60px 20px", color: t.textMuted }}>
                <div style={{ fontFamily: "'Lora', serif", fontSize: "1.1rem", color: t.textSub, marginBottom: 8 }}>Ready to analyse</div>
                <div style={{ fontSize: "0.85rem" }}>Click one of the buttons above to run your models.</div>
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

  const [page, setPage]           = useState("chat"); // "chat" | "dashboard"
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
    setMessages(prev => [...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", raw: null, typing: true },
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
      setMessages(prev => [...prev.slice(0, -1), { role: "assistant", content: data.answer, raw: data.raw_output, typing: false }]);
    } catch (err) {
      setMessages(prev => [...prev.slice(0, -1), { role: "assistant", content: "Something went wrong: " + err.message, raw: null, typing: false }]);
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

  // ── Dashboard page ───────────────────────────────────────────────────────────
  if (page === "dashboard") {
    return (
      <>
        <Dashboard t={t} sessionId={sessionId} onBack={() => setPage("chat")} mode={mode} />
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
      </>
    );
  }

  // ── Chat page ────────────────────────────────────────────────────────────────
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
            {/* Dashboard button */}
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

            {/* Theme toggle */}
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