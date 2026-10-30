import { useMemo, useState } from "react";
import type { Architecture, ModuleNode } from "../types";

interface Props {
  architecture: Architecture | null;
  selectedId: string | null;
  onSelect: (moduleId: string) => void;
  loading: boolean;
}

type ViewMode = "tree" | "layers";

function formatParams(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

function matches(node: ModuleNode, query: string): boolean {
  if (!query) return true;
  return (
    node.id.toLowerCase().includes(query) ||
    node.class_name.toLowerCase().includes(query) ||
    node.name.toLowerCase().includes(query)
  );
}

function subtreeMatches(node: ModuleNode, query: string): boolean {
  return matches(node, query) || node.children.some((child) => subtreeMatches(child, query));
}

function TreeRow({
  node,
  depth,
  query,
  selectedId,
  expanded,
  toggle,
  onSelect,
}: {
  node: ModuleNode;
  depth: number;
  query: string;
  selectedId: string | null;
  expanded: Set<string>;
  toggle: (id: string) => void;
  onSelect: (id: string) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const isExpanded = expanded.has(node.id) || query !== "";
  const hasChildren = node.children.length > 0;
  const visibleChildren = isExpanded ? node.children.filter((child) => subtreeMatches(child, query)) : [];

  return (
    <div>
      <div
        className={`tree-row ${selectedId === node.id ? "selected" : ""}`}
        style={{ paddingLeft: `${depth * 14 + 8}px` }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={() => onSelect(node.id)}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") onSelect(node.id);
        }}
      >
        <button
          className={`tree-toggle ${hasChildren ? "" : "invisible"}`}
          aria-label={isExpanded ? "Collapse" : "Expand"}
          onClick={(event) => {
            event.stopPropagation();
            toggle(node.id);
          }}
        >
          {isExpanded ? "▾" : "▸"}
        </button>
        <span className={`dot ${node.supports_steering ? "steerable" : ""}`} title={node.steering_reason ?? ""} />
        <span className="tree-name">{node.name}</span>
        <span className="tree-class">{node.class_name}</span>
        {node.layer_type && <span className={`badge layer-${node.layer_type}`}>{node.layer_type}</span>}
        {(hovered || selectedId === node.id) && <span className="tree-params">{formatParams(node.parameter_count)}</span>}
      </div>
      {visibleChildren.map((child) => (
        <TreeRow
          key={child.id}
          node={child}
          depth={depth + 1}
          query={query}
          selectedId={selectedId}
          expanded={expanded}
          toggle={toggle}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}

export function ArchitectureExplorer({ architecture, selectedId, onSelect, loading }: Props) {
  const [view, setView] = useState<ViewMode>("tree");
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(
    () => new Set(["root", "model", "model.layers"]),
  );

  const normalizedQuery = query.trim().toLowerCase();
  const root = architecture?.tree;

  const visibleRoot = useMemo(() => {
    if (!root) return null;
    if (!normalizedQuery) return root;
    return subtreeMatches(root, normalizedQuery) ? root : null;
  }, [root, normalizedQuery]);

  const toggle = (id: string) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <section className="panel explorer">
      <header className="panel-header">
        <div>
          <h2>Architecture</h2>
          {architecture && (
            <p className="subtle">
              {architecture.model_class} · {architecture.num_hidden_layers} layers · hidden {architecture.hidden_size}
            </p>
          )}
        </div>
        <div className="segmented">
          <button className={view === "tree" ? "active" : ""} onClick={() => setView("tree")}>
            Tree
          </button>
          <button className={view === "layers" ? "active" : ""} onClick={() => setView("layers")}>
            Layers
          </button>
        </div>
      </header>

      {!architecture && (
        <div className="empty">
          {loading ? "Loading architecture…" : "Load the model to discover its real module hierarchy."}
        </div>
      )}

      {architecture && view === "tree" && (
        <>
          <input
            className="search"
            placeholder="Search module, class, or path"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <div className="tree-scroll">
            {visibleRoot ? (
              <TreeRow
                node={visibleRoot}
                depth={0}
                query={normalizedQuery}
                selectedId={selectedId}
                expanded={expanded}
                toggle={toggle}
                onSelect={onSelect}
              />
            ) : (
              <div className="empty">No modules match “{query}”.</div>
            )}
          </div>
          <p className="legend">
            <span className="dot steerable" /> steerable output · values are real parameter counts
          </p>
        </>
      )}

      {architecture && view === "layers" && (
        <div className="layer-grid">
          {architecture.layer_overview.map((block) => (
            <button
              key={block.module_id}
              className={`layer-card ${selectedId === block.module_id ? "selected" : ""}`}
              onClick={() => onSelect(block.module_id)}
            >
              <div className="layer-card-top">
                <span className="layer-index">{block.layer_index}</span>
                <span className={`badge layer-${block.layer_type}`}>{block.layer_type}</span>
              </div>
              <div className="layer-operator">{block.operator_class}</div>
              <div className="layer-meta">
                {block.operator_type} + MLP · {formatParams(block.parameter_count)} params
              </div>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
