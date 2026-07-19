/**
 * IndeterminateCheckbox.tsx — Native checkbox with programmatic indeterminate state
 * ──────────────────────────────────────────────────────────────────────────────────
 *
 * WHY THIS COMPONENT EXISTS
 * ──────────────────────────
 * React does not map the `indeterminate` JSX prop to the DOM property of the
 * same name — it is simply ignored.  The indeterminate state is write-only
 * (no HTML attribute, only an IDL property) and must be set imperatively via
 * a ref.  This component wraps that imperative call in a useEffect that fires
 * after every render where the prop may have changed, keeping the DOM in sync.
 *
 * USAGE
 * ──────
 * <IndeterminateCheckbox
 *   id="group-gm"
 *   checked={macroState === 'all'}
 *   indeterminate={macroState === 'partial'}
 *   onChange={(checked) => setGroupVisible('gm', checked)}
 * />
 *
 * CLICK SEMANTICS
 * ────────────────
 * The browser fires onChange with e.target.checked reflecting the state
 * the checkbox transitions TO:
 *   • indeterminate → true  (first click always checks)
 *   • checked       → false
 *   • unchecked     → true
 * Callers receive this value directly via the onChange prop.
 */

import { useRef, useEffect, type FC } from 'react';

interface IndeterminateCheckboxProps {
  /** Links to a <label htmlFor={id}> in the parent. */
  id:            string;
  /**
   * Controlled checked state.  When `indeterminate` is true this value is
   * ignored visually, but it is still the value passed through to onChange
   * on the next click (which will be `true` — browsers treat the
   * indeterminate→checked transition as a check, not a toggle of `checked`).
   */
  checked:       boolean;
  /** Show the native browser dash / mixed indicator. */
  indeterminate: boolean;
  /** Receives the value e.target.checked would have after the click. */
  onChange:      (checked: boolean) => void;
  disabled?:     boolean;
}

const IndeterminateCheckbox: FC<IndeterminateCheckboxProps> = ({
  id, checked, indeterminate, onChange, disabled = false,
}) => {
  const ref = useRef<HTMLInputElement>(null);

  // The .indeterminate property is write-only and not reflected as an attribute.
  // Setting it imperatively after each render is the only reliable approach.
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = indeterminate;
  }, [indeterminate]);

  return (
    <input
      ref={ref}
      id={id}
      type="checkbox"
      checked={checked}
      disabled={disabled}
      onChange={(e) => onChange(e.target.checked)}
    />
  );
};

export default IndeterminateCheckbox;
