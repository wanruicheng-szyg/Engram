"use client";

import { useState } from "react";

// ── constants ──────────────────────────────────────────────────────────────────

const OWNER_SS58 = "5FHGPfixYME1CK3EN5YGJzFvXBgtrEqrCnUPLCbvW4RG233k";
const W = 860, H = 520, CX = 430, CY = 265;
const R_VAL = 130, R_MINER = 252;

// ── types ──────────────────────────────────────────────────────────────────────

export interface NVMiner {
  uid: number;
  hotkey: string | null;
  score: number | null;
  vectors: number | null;
  latency_ms: number | null;
  proof_rate: number | null;
  stake: number | null;
  status: "online" | "offline" | "unknown";
}

export interface NVStats {
  netuid: number;
  hotkey: string | null;
  uid: number | null;
  avg_score: number | null;
}

type Role = "owner" | "validator" | "miner";

interface TooltipInfo {
  cx: number;
  cy: number;
  role: Role;
  lines: { label: string; value: string }[];
}

// ── helpers ────────────────────────────────────────────────────────────────────

function scoreColors(score: number | null): { fill: string; stroke: string; glow: string } {
  const s = score ?? 0;
  if (s >= 0.8) return { fill: "#34d399", stroke: "#059669", glow: "rgba(52,211,153,0.55)" };
  if (s >= 0.5) return { fill: "#818cf8", stroke: "#4f46e5", glow: "rgba(129,140,248,0.50)" };
  if (s >= 0.2) return { fill: "#c084fc", stroke: "#9333ea", glow: "rgba(192,132,252,0.40)" };
  return { fill: "#374151", stroke: "#1f2937", glow: "rgba(55,65,81,0.25)" };
}

function roleAccent(role: Role) {
  return role === "owner" ? "#fbbf24" : role === "validator" ? "#a855f7" : "#818cf8";
}

// ── tooltip rendered inside SVG ────────────────────────────────────────────────

function Tooltip({ info }: { info: TooltipInfo }) {
  const PAD = 10, LINE = 15, HEADER_H = 24;
  const boxW = 168;
  const boxH = HEADER_H + info.lines.length * LINE + PAD;
  // keep box inside viewBox
  const x = info.cx + 20 + boxW > W ? info.cx - boxW - 14 : info.cx + 14;
  const y = Math.max(6, Math.min(info.cy - 20, H - boxH - 6));
  const accent = roleAccent(info.role);
  return (
    <g style={{ pointerEvents: "none" }}>
      {/* drop shadow */}
      <rect x={x + 2} y={y + 2} width={boxW} height={boxH} rx={8}
        fill="rgba(0,0,0,0.6)" />
      <rect x={x} y={y} width={boxW} height={boxH} rx={8}
        fill="#09070e" stroke="rgba(255,255,255,0.09)" strokeWidth="0.8" />
      {/* role label */}
      <text x={x + PAD} y={y + 16} fontSize="9" fontFamily="monospace"
        fontWeight="bold" fill={accent} letterSpacing="2.5">
        {info.role.toUpperCase()}
      </text>
      {/* detail rows */}
      {info.lines.map(({ label, value }, i) => (
        <g key={label}>
          <text x={x + PAD} y={y + HEADER_H + i * LINE + 11}
            fontSize="8.5" fontFamily="monospace" fill="rgba(255,255,255,0.28)">{label}</text>
          <text x={x + PAD + 56} y={y + HEADER_H + i * LINE + 11}
            fontSize="8.5" fontFamily="monospace" fill="rgba(255,255,255,0.82)">{value}</text>
        </g>
      ))}
    </g>
  );
}

// ── view toggle ────────────────────────────────────────────────────────────────

type ViewMode = "network" | "heatmap";

// ── network view ───────────────────────────────────────────────────────────────

function NetworkView({
  miners,
  stats,
  onHover,
  hovered,
}: {
  miners: NVMiner[];
  stats: NVStats | null;
  onHover: (info: TooltipInfo | null) => void;
  hovered: TooltipInfo | null;
}) {
  const hasVal = Boolean(stats?.hotkey);

  const minerNodes = miners.map((m, i) => {
    const angle = (2 * Math.PI * i) / Math.max(miners.length, 1) - Math.PI / 2;
    const r = Math.max(9, Math.min(22, 9 + (m.score ?? 0) * 13));
    return {
      m,
      cx: CX + Math.cos(angle) * R_MINER,
      cy: CY + Math.sin(angle) * R_MINER,
      r,
      pulseDur: `${2.0 + (m.uid % 15) / 10}s`,
      ...scoreColors(m.score),
    };
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto"
      onMouseLeave={() => onHover(null)}>
      <defs>
        <radialGradient id="nv-bg" cx="50%" cy="50%" r="65%">
          <stop offset="0%" stopColor="#0d0b14" />
          <stop offset="100%" stopColor="#080608" />
        </radialGradient>
        <radialGradient id="nv-owner" cx="32%" cy="28%">
          <stop offset="0%" stopColor="#fde68a" />
          <stop offset="100%" stopColor="#92400e" />
        </radialGradient>
        <radialGradient id="nv-val" cx="32%" cy="28%">
          <stop offset="0%" stopColor="#d8b4fe" />
          <stop offset="100%" stopColor="#4c1d95" />
        </radialGradient>

        <filter id="nv-f-owner" x="-120%" y="-120%" width="340%" height="340%">
          <feGaussianBlur stdDeviation="12" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="nv-f-val" x="-90%" y="-90%" width="280%" height="280%">
          <feGaussianBlur stdDeviation="7" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="nv-f-miner" x="-70%" y="-70%" width="240%" height="240%">
          <feGaussianBlur stdDeviation="4" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>

        <pattern id="nv-grid" width="48" height="48" patternUnits="userSpaceOnUse">
          <path d="M 48 0 L 0 0 0 48" fill="none"
            stroke="rgba(255,255,255,0.018)" strokeWidth="0.5" />
        </pattern>
      </defs>

      {/* bg */}
      <rect width={W} height={H} fill="url(#nv-bg)" />
      <rect width={W} height={H} fill="url(#nv-grid)" />

      {/* orbit rings */}
      <circle cx={CX} cy={CY} r={R_VAL} fill="none"
        stroke="rgba(168,85,247,0.06)" strokeWidth="0.8" strokeDasharray="3 7" />
      <circle cx={CX} cy={CY} r={R_MINER} fill="none"
        stroke="rgba(99,102,241,0.05)" strokeWidth="0.8" strokeDasharray="3 9" />

      {/* edge: owner → validator */}
      {hasVal && (
        <line x1={CX} y1={CY} x2={CX} y2={CY - R_VAL}
          stroke="rgba(251,191,36,0.22)" strokeWidth="1.5" strokeDasharray="5 5">
          <animate attributeName="stroke-dashoffset" from="0" to="-10"
            dur="1.8s" repeatCount="indefinite" />
        </line>
      )}

      {/* edges: (validator or owner) → miners */}
      {minerNodes.map((n) => (
        <line key={`e-${n.m.uid}`}
          x1={hasVal ? CX : CX} y1={hasVal ? CY - R_VAL : CY}
          x2={n.cx} y2={n.cy}
          stroke={n.m.status === "online"
            ? "rgba(99,102,241,0.11)"
            : "rgba(99,102,241,0.03)"}
          strokeWidth="0.8" />
      ))}

      {/* miner nodes */}
      {minerNodes.map((n) => (
        <g key={`mn-${n.m.uid}`} style={{ cursor: "pointer" }}
          onMouseEnter={() => onHover({
            cx: n.cx, cy: n.cy, role: "miner",
            lines: [
              { label: "UID", value: String(n.m.uid) },
              { label: "Score", value: n.m.score != null ? n.m.score.toFixed(4) : "—" },
              { label: "Vectors", value: n.m.vectors != null ? n.m.vectors.toLocaleString() : "—" },
              { label: "Latency", value: n.m.latency_ms != null ? `${n.m.latency_ms}ms` : "—" },
              { label: "Proof", value: n.m.proof_rate != null ? `${(n.m.proof_rate * 100).toFixed(0)}%` : "—" },
              { label: "Status", value: n.m.status },
            ],
          })}>
          {n.m.status === "online" && (
            <circle cx={n.cx} cy={n.cy} r={n.r + 4} fill="none"
              stroke={n.glow} strokeWidth="1">
              <animate attributeName="r"
                values={`${n.r + 4};${n.r + 11};${n.r + 4}`}
                dur={n.pulseDur} repeatCount="indefinite" />
              <animate attributeName="stroke-opacity" values="0.7;0;0.7"
                dur={n.pulseDur} repeatCount="indefinite" />
            </circle>
          )}
          <circle cx={n.cx} cy={n.cy} r={n.r}
            fill={n.fill} stroke={n.stroke} strokeWidth="1.2"
            opacity={n.m.status === "online" ? 1 : 0.28}
            filter={n.m.status === "online" ? "url(#nv-f-miner)" : undefined} />
          {n.r >= 11 && (
            <text x={n.cx} y={n.cy} textAnchor="middle" dominantBaseline="middle"
              fontSize="7.5" fontFamily="monospace" fill="rgba(255,255,255,0.88)"
              fontWeight="bold">
              {n.m.uid}
            </text>
          )}
        </g>
      ))}

      {/* validator node */}
      {hasVal && (
        <g style={{ cursor: "pointer" }}
          onMouseEnter={() => onHover({
            cx: CX, cy: CY - R_VAL, role: "validator",
            lines: [
              { label: "UID", value: String(stats?.uid ?? "—") },
              { label: "Hotkey", value: stats?.hotkey
                ? `${stats.hotkey.slice(0, 8)}…${stats.hotkey.slice(-4)}` : "—" },
              { label: "Avg Score", value: stats?.avg_score != null
                ? stats.avg_score.toFixed(4) : "—" },
            ],
          })}>
          <circle cx={CX} cy={CY - R_VAL} r={28} fill="none"
            stroke="rgba(168,85,247,0.22)" strokeWidth="1">
            <animate attributeName="r" values="28;42;28" dur="2.8s" repeatCount="indefinite" />
            <animate attributeName="stroke-opacity" values="0.22;0;0.22"
              dur="2.8s" repeatCount="indefinite" />
          </circle>
          <circle cx={CX} cy={CY - R_VAL} r={22}
            fill="url(#nv-val)" stroke="#7c3aed" strokeWidth="1.5"
            filter="url(#nv-f-val)" />
          <text x={CX} y={CY - R_VAL} textAnchor="middle" dominantBaseline="middle"
            fontSize="8.5" fontFamily="monospace" fill="#e9d5ff" fontWeight="bold">
            VAL
          </text>
        </g>
      )}

      {/* owner node */}
      <g style={{ cursor: "pointer" }}
        onMouseEnter={() => onHover({
          cx: CX, cy: CY, role: "owner",
          lines: [
            { label: "Role", value: "Subnet Owner" },
            { label: "Address", value: `${OWNER_SS58.slice(0, 10)}…${OWNER_SS58.slice(-6)}` },
            { label: "Subnet", value: "netuid 450" },
            { label: "Network", value: "Bittensor testnet" },
          ],
        })}>
        {[0, 1].map((i) => (
          <circle key={i} cx={CX} cy={CY} r={44 + i * 9} fill="none"
            stroke="rgba(251,191,36,0.14)" strokeWidth="1">
            <animate attributeName="r"
              values={`${44 + i * 9};${62 + i * 14};${44 + i * 9}`}
              dur={`${3.6 + i * 0.9}s`} repeatCount="indefinite"
              begin={`${i * 0.9}s`} />
            <animate attributeName="stroke-opacity" values="0.14;0;0.14"
              dur={`${3.6 + i * 0.9}s`} repeatCount="indefinite"
              begin={`${i * 0.9}s`} />
          </circle>
        ))}
        <circle cx={CX} cy={CY} r={40}
          fill="url(#nv-owner)" stroke="#f59e0b" strokeWidth="2"
          filter="url(#nv-f-owner)" />
        <text x={CX} y={CY - 5} textAnchor="middle" dominantBaseline="middle"
          fontSize="9" fontFamily="monospace" fill="#fde68a" fontWeight="bold">
          OWNER
        </text>
        <text x={CX} y={CY + 8} textAnchor="middle" dominantBaseline="middle"
          fontSize="7" fontFamily="monospace" fill="rgba(253,230,138,0.45)">
          netuid 450
        </text>
      </g>

      {/* tooltip */}
      {hovered && <Tooltip info={hovered} />}

      {/* bottom-right status */}
      <text x={W - 14} y={H - 10} textAnchor="end"
        fontSize="9" fontFamily="monospace" fill="rgba(255,255,255,0.12)">
        {miners.filter((m) => m.status === "online").length}/{miners.length} online
      </text>
    </svg>
  );
}

// ── heatmap view ───────────────────────────────────────────────────────────────

function HeatmapView({
  miners,
  onHover,
  hovered,
}: {
  miners: NVMiner[];
  onHover: (info: TooltipInfo | null) => void;
  hovered: TooltipInfo | null;
}) {
  const COLS = Math.ceil(Math.sqrt(Math.max(miners.length, 1) * (W / H)));
  const ROWS = Math.ceil(miners.length / COLS);
  const CELL = Math.min(Math.floor(W / COLS), Math.floor(H / ROWS)) - 4;
  const GAP = 4;
  const STEP = CELL + GAP;
  const startX = (W - COLS * STEP + GAP) / 2;
  const startY = (H - ROWS * STEP + GAP) / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto"
      onMouseLeave={() => onHover(null)}>
      <defs>
        <pattern id="hm-grid" width="48" height="48" patternUnits="userSpaceOnUse">
          <path d="M 48 0 L 0 0 0 48" fill="none"
            stroke="rgba(255,255,255,0.018)" strokeWidth="0.5" />
        </pattern>
      </defs>
      <rect width={W} height={H} fill="#080608" />
      <rect width={W} height={H} fill="url(#hm-grid)" />

      {miners.map((m, i) => {
        const col = i % COLS;
        const row = Math.floor(i / COLS);
        const x = startX + col * STEP;
        const y = startY + row * STEP;
        const { fill, stroke } = scoreColors(m.score);
        const s = m.score ?? 0;
        const opacity = m.status === "online" ? 0.2 + s * 0.8 : 0.12;
        return (
          <g key={m.uid} style={{ cursor: "pointer" }}
            onMouseEnter={() => onHover({
              cx: x + CELL / 2, cy: y,
              role: "miner",
              lines: [
                { label: "UID", value: String(m.uid) },
                { label: "Score", value: m.score != null ? m.score.toFixed(4) : "—" },
                { label: "Vectors", value: m.vectors != null ? m.vectors.toLocaleString() : "—" },
                { label: "Latency", value: m.latency_ms != null ? `${m.latency_ms}ms` : "—" },
                { label: "Proof", value: m.proof_rate != null ? `${(m.proof_rate * 100).toFixed(0)}%` : "—" },
                { label: "Status", value: m.status },
              ],
            })}>
            <rect x={x} y={y} width={CELL} height={CELL} rx={6}
              fill={fill} stroke={stroke} strokeWidth="0.8" opacity={opacity} />
            {CELL >= 36 && (
              <text x={x + CELL / 2} y={y + CELL / 2 - 4}
                textAnchor="middle" dominantBaseline="middle"
                fontSize={CELL >= 50 ? "11" : "9"} fontFamily="monospace"
                fill="rgba(255,255,255,0.85)" fontWeight="bold">
                {m.uid}
              </text>
            )}
            {CELL >= 50 && m.score != null && (
              <text x={x + CELL / 2} y={y + CELL / 2 + 9}
                textAnchor="middle" dominantBaseline="middle"
                fontSize="8" fontFamily="monospace" fill="rgba(255,255,255,0.45)">
                {m.score.toFixed(2)}
              </text>
            )}
          </g>
        );
      })}

      {miners.length === 0 && (
        <text x={W / 2} y={H / 2} textAnchor="middle" dominantBaseline="middle"
          fontSize="13" fontFamily="monospace" fill="rgba(255,255,255,0.18)">
          no miners registered yet
        </text>
      )}

      {hovered && <Tooltip info={hovered} />}
    </svg>
  );
}

// ── main export ────────────────────────────────────────────────────────────────

export default function NetworkVisualizer({
  miners,
  stats,
}: {
  miners: NVMiner[];
  stats: NVStats | null;
}) {
  const [view, setView] = useState<ViewMode>("network");
  const [hovered, setHovered] = useState<TooltipInfo | null>(null);

  const onlineCount = miners.filter((m) => m.status === "online").length;

  return (
    <div className="rounded-xl overflow-hidden border border-white/[0.14] bg-[#080608]">
      {/* header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.12] bg-[#0d0b11]">
        <div>
          <h2 className="font-display font-semibold text-[18px] text-white leading-tight">
            Network Map
          </h2>
          <p className="text-[11px] font-mono text-white/55 mt-0.5">
            {view === "network"
              ? "owner → validator → miners · hover to inspect"
              : "miners by score intensity · hover to inspect"}
          </p>
        </div>

        <div className="flex items-center gap-4">
          {/* legend */}
          <div className="hidden sm:flex items-center gap-3 text-[10px] font-mono text-white/55 uppercase tracking-widest">
            {view === "network" ? (
              <>
                {[
                  { color: "#fbbf24", label: "owner" },
                  { color: "#a855f7", label: "validator" },
                  { color: "#34d399", label: "miner ≥0.8" },
                  { color: "#818cf8", label: "0.5–0.8" },
                  { color: "#c084fc", label: "0.2–0.5" },
                ].map(({ color, label }) => (
                  <span key={label} className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: color }} />
                    {label}
                  </span>
                ))}
              </>
            ) : (
              <span className="text-white/20">
                {onlineCount}/{miners.length} online
              </span>
            )}
          </div>

          {/* view toggle */}
          <div className="flex rounded-lg overflow-hidden border border-white/[0.14]">
            {(["network", "heatmap"] as ViewMode[]).map((v) => (
              <button key={v}
                onClick={() => { setView(v); setHovered(null); }}
                className={`px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest transition-colors ${
                  view === v
                    ? "bg-[#7c3aed]/20 text-[#a855f7]"
                    : "bg-transparent text-white/55 hover:text-white/50"
                }`}>
                {v}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* visualizer */}
      {view === "network" ? (
        <NetworkView miners={miners} stats={stats} onHover={setHovered} hovered={hovered} />
      ) : (
        <HeatmapView miners={miners} onHover={setHovered} hovered={hovered} />
      )}
    </div>
  );
}
