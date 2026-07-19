/**
 * NirsViewer.tsx — fNIRS multi-channel waveform viewer
 * ──────────────────────────────────────────────────────
 *
 * Renders SNIRF (fNIRS) data as a stacked timeseries display, reusing the
 * canvas renderer from the timeseries plugin. fNIRS channels are labelled
 * by source-detector pair and wavelength.
 *
 * The viewer owns NirsContext (consumed by NirsControls), which follows the
 * same pattern as TimeseriesContext in TimeseriesViewer.tsx.
 *
 * The viewer converts SnirfPayload → a display-ready data structure, then
 * delegates rendering to useCanvasRenderer (the same canvas hook used by
 * TimeseriesViewer).
 */

import {
  type FC,
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type RefObject,
} from 'react';
import type { PluginViewerProps } from '../../types/plugin.types';
import type { SnirfPayload, TimeseriesViewerControls } from '../../types/timeseries.types';
import { useCanvasRenderer } from '../timeseries/hooks/useCanvasRenderer';
import type { TimeseriesPayload } from '../../types/timeseries.types';

// ── Context ───────────────────────────────────────────────────────────────────

export interface NirsContextValue {
  snirfPayload: SnirfPayload | null;
  controls:     TimeseriesViewerControls;
  setControls:  (partial: Partial<TimeseriesViewerControls>) => void;
  totalDuration: number;
  selectedWavelengthIdx: number;
  setSelectedWavelengthIdx: (idx: number) => void;
}

export const NirsContext = createContext<NirsContextValue | null>(null);

export function useNirsContext(): NirsContextValue {
  const ctx = useContext(NirsContext);
  if (!ctx) throw new Error('useNirsContext must be used inside NirsViewer');
  return ctx;
}

// ── SNIRF → display converter ─────────────────────────────────────────────────

function snirfToTimeseries(
  payload: SnirfPayload,
  wavelengthIdx: number, // 1-based, matches SnirfMeasurementListEntry.wavelengthIndex
): TimeseriesPayload {
  const { dataTimeSeries, numTimePoints, numChannels, time, measurementList, wavelengths, sourceLabels, detectorLabels } = payload;

  // Filter channels that match the selected wavelength
  const filteredIndices = measurementList
    .map((ml, i) => ({ ml, i }))
    .filter(({ ml }) => ml.wavelengthIndex === wavelengthIdx);

  const wlNm = wavelengths[wavelengthIdx - 1];
  const wlLabel = wlNm !== undefined ? `${Math.round(wlNm)} nm` : `λ${wavelengthIdx}`;

  const channels = filteredIndices.map(({ ml }) => {
    const srcLabel = sourceLabels[ml.sourceIndex - 1] ?? `S${ml.sourceIndex}`;
    const detLabel = detectorLabels[ml.detectorIndex - 1] ?? `D${ml.detectorIndex}`;
    return {
      label:      `${srcLabel}-${detLabel} ${wlLabel}`,
      unit:       'OD',
      sampleRate: numTimePoints > 0 && time[numTimePoints - 1]! > 0
        ? numTimePoints / time[numTimePoints - 1]!
        : 0,
      visible: true,
      type:    'NIRS',
    };
  });

  // Extract the channel columns from the row-major flat matrix
  const data: Float32Array[] = filteredIndices.map(({ i: chIdx }) => {
    const col = new Float32Array(numTimePoints);
    for (let t = 0; t < numTimePoints; t++) {
      col[t] = dataTimeSeries[t * numChannels + chIdx]!;
    }
    return col;
  });

  return {
    time,
    numSamples: numTimePoints,
    channels,
    data,
    sourceModality: 'nirs',
    filename: '',
  };
}

// ── Default controls ──────────────────────────────────────────────────────────

function defaultControls(payload: SnirfPayload | null): TimeseriesViewerControls {
  const n    = payload?.numTimePoints ?? 0;
  const last = payload ? payload.time[n - 1]! : 30;
  return {
    timeStart:           0,
    timeEnd:             Math.min(60, last),
    amplitudeScale:      0.01,
    laneHeightPx:        60,
    firstVisibleChannel: 0,
    visibleChannelCount: 20,
    selectedChannels:    Array.from({ length: Math.min(20, payload?.numChannels ?? 0) }, (_, i) => i),
    filterLowHz:         0,
    filterHighHz:        0,
  };
}

// ── Component ─────────────────────────────────────────────────────────────────

const NirsViewer: FC<PluginViewerProps> = ({ data, controlsSlot }) => {
  const snirfPayload = data?.kind === 'nirs' ? data.payload : null;

  const [controls, setControlsState] = useState<TimeseriesViewerControls>(
    () => defaultControls(snirfPayload),
  );
  const [selectedWavelengthIdx, setSelectedWavelengthIdx] = useState(1);

  const setControls = useCallback(
    (partial: Partial<TimeseriesViewerControls>) =>
      setControlsState((prev) => ({ ...prev, ...partial })),
    [],
  );

  useEffect(() => {
    setControlsState(defaultControls(snirfPayload));
    setSelectedWavelengthIdx(1);
  }, [snirfPayload]);

  const totalDuration = snirfPayload
    ? snirfPayload.time[snirfPayload.numTimePoints - 1]! - snirfPayload.time[0]!
    : 0;

  // Convert SnirfPayload to TimeseriesPayload for the canvas renderer
  const tsPayload: TimeseriesPayload | null = snirfPayload
    ? snirfToTimeseries(snirfPayload, selectedWavelengthIdx)
    : null;

  // Filter selected channels by what's visible after wavelength filtering
  const filteredControls: TimeseriesViewerControls = {
    ...controls,
    selectedChannels: tsPayload
      ? tsPayload.channels.map((_, i) => i).slice(0, controls.visibleChannelCount)
      : [],
  };

  const canvasRef = useCanvasRenderer(tsPayload, filteredControls);

  const contextValue: NirsContextValue = {
    snirfPayload,
    controls,
    setControls,
    totalDuration,
    selectedWavelengthIdx,
    setSelectedWavelengthIdx,
  };

  return (
    <NirsContext.Provider value={contextValue}>
      <div className="plugin-workspace">
        <div className="ts-container" tabIndex={0} style={{ outline: 'none' }}>
          {snirfPayload ? (
            <>
              <canvas
                ref={canvasRef as RefObject<HTMLCanvasElement>}
                className="ts-canvas"
                style={{ width: '100%', height: '100%', display: 'block' }}
              />
              <NirsScrollbar snirfPayload={snirfPayload} controls={controls} />
            </>
          ) : (
            <div className="empty-state">
              <p>Upload a SNIRF fNIRS file (.snirf) to begin</p>
            </div>
          )}
        </div>
        {controlsSlot}
      </div>
    </NirsContext.Provider>
  );
};

// ── Scrollbar ─────────────────────────────────────────────────────────────────

function NirsScrollbar({ controls }: { snirfPayload: SnirfPayload; controls: TimeseriesViewerControls }) {
  const { setControls, totalDuration } = useNirsContext();
  if (totalDuration <= 0) return null;

  const thumbLeft  = controls.timeStart / totalDuration;
  const thumbWidth = (controls.timeEnd - controls.timeStart) / totalDuration;

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const bar = e.currentTarget.getBoundingClientRect();
    const relX = (e.clientX - bar.left) / bar.width;
    const newCentre = relX * totalDuration;
    const halfWin   = (controls.timeEnd - controls.timeStart) / 2;
    setControls({
      timeStart: Math.max(0, newCentre - halfWin),
      timeEnd:   Math.min(totalDuration, newCentre + halfWin),
    });
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  return (
    <div className="ts-scrollbar" onPointerDown={onPointerDown}
      role="scrollbar" aria-orientation="horizontal">
      <div
        className="ts-scrollbar__thumb"
        style={{ left: `${(thumbLeft * 100).toFixed(2)}%`, width: `${(thumbWidth * 100).toFixed(2)}%` }}
      />
    </div>
  );
}

export default NirsViewer;
