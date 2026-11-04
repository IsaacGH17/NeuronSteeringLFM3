import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

export const TOOLTIPS_STORAGE_KEY = "lfm3.tooltips.enabled";

interface TooltipContextValue {
  enabled: boolean;
  setEnabled: (enabled: boolean) => void;
}

const TooltipContext = createContext<TooltipContextValue>({
  enabled: true,
  setEnabled: () => undefined,
});

function readStoredPreference(): boolean {
  try {
    const raw = window.localStorage.getItem(TOOLTIPS_STORAGE_KEY);
    if (raw === "false") return false;
    if (raw === "true") return true;
  } catch {
    // localStorage can be unavailable (private mode, restricted webviews)
  }
  return true;
}

export function TooltipProvider({ children }: { children: ReactNode }) {
  const [enabled, setEnabled] = useState(readStoredPreference);

  useEffect(() => {
    try {
      window.localStorage.setItem(TOOLTIPS_STORAGE_KEY, String(enabled));
    } catch {
      // preference stays in memory when storage is unavailable
    }
  }, [enabled]);

  const value = useMemo(() => ({ enabled, setEnabled }), [enabled]);
  return <TooltipContext.Provider value={value}>{children}</TooltipContext.Provider>;
}

export function useTooltips(): TooltipContextValue {
  return useContext(TooltipContext);
}

export function tooltipTitle(enabled: boolean, text: string | null | undefined): string | undefined {
  return enabled && text ? text : undefined;
}
