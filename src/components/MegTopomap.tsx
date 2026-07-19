import { type FC, useEffect, useRef, useState } from 'react';
import { megApi } from '../services/megApi';

interface MegTopomapProps {
  sessionId: string | undefined;
  tStart:    number;
  tEnd:      number;
}

const MegTopomap: FC<MegTopomapProps> = ({ sessionId, tStart, tEnd }) => {
  const [image,   setImage]   = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    const timer = setTimeout(async () => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      setLoading(true);
      setError(null);
      try {
        const res = await megApi.getTopomap(sessionId, tStart, tEnd, ctrl.signal);
        setImage(res.image);
      } catch (err: unknown) {
        if ((err as { name?: string }).name === 'AbortError') return;
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    }, 600);

    return () => {
      clearTimeout(timer);
      abortRef.current?.abort();
    };
  }, [sessionId, tStart, tEnd]);

  if (!sessionId) return null;

  return (
    <div style={{ textAlign: 'center', marginTop: 4 }}>
      {loading && (
        <div style={{ color: '#9a9ab8', fontSize: 10, padding: '12px 0' }}>
          Rendering…
        </div>
      )}
      {error && (
        <p style={{ color: 'var(--accent-red, #e05252)', fontSize: 9, margin: '4px 0' }}>
          {error}
        </p>
      )}
      {image && !loading && (
        <img
          src={image}
          alt="MEG topomap"
          style={{ width: '100%', maxWidth: 220, borderRadius: 6, display: 'block', margin: '0 auto' }}
        />
      )}
    </div>
  );
};

export default MegTopomap;
