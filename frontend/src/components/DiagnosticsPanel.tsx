import { useState } from "react";
import type { Diagnostics } from "../types";

function formatBytes(value: number | null): string {
  if (value === null) return "—";
  const gib = value / 1024 ** 3;
  return `${gib.toFixed(2)} GiB`;
}

export function DiagnosticsPanel({ diagnostics }: { diagnostics: Diagnostics | null }) {
  const [open, setOpen] = useState(false);

  return (
    <section className={`diagnostics ${open ? "open" : ""}`}>
      <button className="diagnostics-toggle" onClick={() => setOpen((value) => !value)}>
        <span>{open ? "▾" : "▸"} Diagnostics & environment</span>
        {diagnostics && (
          <span className="subtle">
            {diagnostics.backend} · {diagnostics.device_name ?? "no accelerator"} · model{" "}
            {diagnostics.model_state}
          </span>
        )}
      </button>
      {open && diagnostics && (
        <div className="diagnostics-body">
          <dl className="meta-grid">
            <div>
              <dt>Python</dt>
              <dd>{diagnostics.python_version}</dd>
            </div>
            <div>
              <dt>PyTorch</dt>
              <dd>{diagnostics.torch_version}</dd>
            </div>
            <div>
              <dt>Transformers</dt>
              <dd>{diagnostics.transformers_version}</dd>
            </div>
            <div>
              <dt>FastAPI</dt>
              <dd>{diagnostics.fastapi_version}</dd>
            </div>
            <div>
              <dt>torch.cuda.is_available()</dt>
              <dd>{String(diagnostics.cuda_available)}</dd>
            </div>
            <div>
              <dt>HIP version</dt>
              <dd>{diagnostics.hip_version ?? "—"}</dd>
            </div>
            <div>
              <dt>Backend</dt>
              <dd>{diagnostics.backend}</dd>
            </div>
            <div>
              <dt>Visible devices</dt>
              <dd>{diagnostics.device_count}</dd>
            </div>
            <div>
              <dt>Device name</dt>
              <dd>{diagnostics.device_name ?? "—"}</dd>
            </div>
            <div>
              <dt>Inference device</dt>
              <dd>{diagnostics.inference_device ?? "—"}</dd>
            </div>
            <div>
              <dt>Allocated VRAM</dt>
              <dd>{formatBytes(diagnostics.memory?.allocated_bytes ?? null)}</dd>
            </div>
            <div>
              <dt>Free VRAM</dt>
              <dd>{formatBytes(diagnostics.memory?.free_bytes ?? null)}</dd>
            </div>
          </dl>
          {diagnostics.notes.map((note) => (
            <p key={note} className="subtle">
              {note}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}
