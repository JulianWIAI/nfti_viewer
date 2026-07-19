/**
 * FrequencyDashboard.tsx — Horizontal bar chart of EEG/MEG frequency band power
 * ──────────────────────────────────────────────────────────────────────────────
 * Displays the relative power (0–100 %) in the five standard neurological
 * frequency bands returned by /api/meg/frequency-bands.  Longer bars indicate
 * greater relative contribution to the 1–50 Hz spectrum.
 *
 * Band colour coding (conventional neuroscience palette):
 *   δ Delta  1–4 Hz   — violet  (dominant in deep sleep / coma)
 *   θ Theta  4–8 Hz   — sky     (drowsiness, hippocampal encoding)
 *   α Alpha  8–12 Hz  — green   (relaxed wakefulness, eyes closed)
 *   β Beta  12–30 Hz  — amber   (active thought, motor)
 *   γ Gamma 30–50 Hz  — coral   (high-level cognition, can be muscle)
 *
 * Implemented with plain CSS — no external charting library.
 */

import { useState, useCallback, type FC } from 'react';
import type { BandPower, FrequencyBandsResult } from '../types/analysis.types';
import { megAnalysisApi } from '../services/megAnalysisApi';

// ── Band configuration ────────────────────────────────────────────────────────

interface BandConfig {
  key:    keyof BandPower;
  label:  string;
  range:  string;
  color:  string;
  hint:   string;
}

const BAND_CONFIG: ReadonlyArray<BandConfig> = [
  { key: 'delta', label: 'δ Delta',  range: '1–4 Hz',   color: '#b39ddb', hint: 'Deep sleep / large lesion' },
  { key: 'theta', label: 'θ Theta',  range: '4–8 Hz',   color: '#81d4fa', hint: 'Drowsiness / memory' },
  { key: 'alpha', label: 'α Alpha',  range: '8–12 Hz',  color: '#a5d6a7', hint: 'Relaxed wakefulness' },
  { key: 'beta',  label: 'β Beta',   range: '12–30 Hz', color: '#ffcc80', hint: 'Active cognition / motor' },
  { key: 'gamma', label: 'γ Gamma',  range: '30–50 Hz', color: '#ef9a9a', hint: 'High cognition / muscle' },
];

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  /**
   * Pre-fetched band-power result.  When provided the component is "display only".
   * When omitted the component renders a "Compute" button and manages its own
   * fetch state (requires sessionId).
   */
  result?:    FrequencyBandsResult;
  /** MEG session ID — required when result is not pre-fetched. */
  sessionId?: string;
  /** Optional time-window restriction forwarded to the backend. */
  tStart?:    number;
  tEnd?:      number;
}

const FrequencyDashboard: FC<Props> = ({ result: initialResult, sessionId, tStart, tEnd }) => {
  const [result,  setResult]  = useState<FrequencyBandsResult | null>(initialResult ?? null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  // Fetch band power on demand when the result was not passed as a prop.
  const handleCompute = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await megAnalysisApi.getFrequencyBands(sessionId, tStart, tEnd);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [sessionId, tStart, tEnd]);

  const bands = result?.bands;
  // Normalise bars relative to the strongest band so the widest bar fills 100%.
  const maxPower = bands ? Math.max(...Object.values(bands)) : 1;

  return (
    <div>
      {/* ── Trigger button (only shown when managed internally) ─────────────── */}
      {!initialResult && (
        <button
          className="btn btn--primary"
          disabled={!sessionId || loading}
          onClick={handleCompute}
          style={{ marginBottom: 8, width: '100%' }}
        >
          {loading ? 'Analysing…' : 'Compute Band Power'}
        </button>
      )}

      {error && (
        <p style={{ color: 'var(--accent-red, #e05252)', fontSize: 10, marginBottom: 6 }}>
          Error: {error}
        </p>
      )}

      {/* ── Band power bars ─────────────────────────────────────────────────── */}
      {bands && BAND_CONFIG.map(({ key, label, range, color, hint }) => {
        const power = bands[key];
        const pct   = (power * 100).toFixed(1);
        // Bar fill width relative to the strongest band (not absolute %).
        const fillPct = `${((power / maxPower) * 100).toFixed(1)}%`;

        return (
          <div key={key} style={{ marginBottom: 10 }}>
            {/* Row header: band name | range | percentage */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 10,
              marginBottom: 3,
              lineHeight: 1,
            }}>
              <span style={{ color, fontWeight: 700, minWidth: 54 }}>{label}</span>
              <span style={{ color: '#666', flex: 1, paddingLeft: 4 }}>{range}</span>
              <span style={{ color: '#b0b0c8', fontVariantNumeric: 'tabular-nums' }}>{pct}%</span>
            </div>

            {/* Bar track */}
            <div style={{
              background: 'rgba(255,255,255,0.05)',
              borderRadius: 3,
              height: 10,
              overflow: 'hidden',
            }}>
              <div style={{
                width:       fillPct,
                height:      '100%',
                background:  color,
                opacity:     0.82,
                borderRadius: 3,
                // Animate when the value changes (e.g. time window update).
                transition:  'width 0.4s ease',
              }} />
            </div>

            {/* Hint text */}
            <div style={{ fontSize: 9, color: '#4a4a6a', marginTop: 2 }}>{hint}</div>
          </div>
        );
      })}

      {/* ── Metadata footer ─────────────────────────────────────────────────── */}
      {result && (
        <div style={{ fontSize: 9, color: '#555', marginTop: 4 }}>
          {result.n_channels} {result.channel_types.join('/')} channels · 1–50 Hz Welch PSD
        </div>
      )}

      {/* ── Empty state ─────────────────────────────────────────────────────── */}
      {!bands && !loading && (
        <p style={{ color: '#555', fontSize: 10, textAlign: 'center', marginTop: 8 }}>
          {sessionId ? 'Click "Compute Band Power" to analyse.' : 'No MEG session loaded.'}
        </p>
      )}
    </div>
  );
};

export default FrequencyDashboard;
