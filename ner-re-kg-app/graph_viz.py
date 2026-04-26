from __future__ import annotations

import json
from typing import Iterable


LABEL_COLOR = {
    "PERSON": "#4A90E2",
    "ORG": "#7ED321",
    "LOC": "#F5A623",
    "UNKNOWN": "#BDBDBD",
}


def build_nodes(entities: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    nodes: list[dict] = []
    for e in entities:
        entity_text = str(e.get("text", "")).strip()
        if not entity_text or entity_text in seen:
            continue
        seen.add(entity_text)
        label = str(e.get("label", "UNKNOWN"))
        color = LABEL_COLOR.get(label, LABEL_COLOR["UNKNOWN"])
        size = 30 if label == "PERSON" else 26 if label == "ORG" else 22 if label == "LOC" else 18
        nodes.append(
            {
                "id": entity_text,
                "label": entity_text,
                "group": label,
                "color": color,
                "size": size,
            }
        )
    return nodes


def build_edges(relations: Iterable[dict]) -> list[dict]:
    edges: list[dict] = []
    for r in relations:
        subj = str(r.get("subject", "")).strip()
        obj = str(r.get("object", "")).strip()
        rel = str(r.get("relation", "")).strip()
        if not subj or not obj or not rel:
            continue
        edges.append({"from": subj, "to": obj, "label": rel, "arrows": "to"})
    return edges


def ensure_nodes_for_relations(nodes: list[dict], edges: list[dict]) -> list[dict]:
    existing = {n["id"] for n in nodes}
    for e in edges:
        for endpoint in (e["from"], e["to"]):
            if endpoint in existing:
                continue
            existing.add(endpoint)
            nodes.append(
                {
                    "id": endpoint,
                    "label": endpoint,
                    "group": "UNKNOWN",
                    "color": LABEL_COLOR["UNKNOWN"],
                    "size": 18,
                }
            )
    return nodes


def render_vis_network_html(nodes: list[dict], edges: list[dict], height_px: int = 600) -> str:
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <script src=\"https://unpkg.com/vis-network/standalone/umd/vis-network.min.js\"></script>
    <style>
      html, body {{ margin: 0; padding: 0; }}
      #mynetwork {{ width: 100%; height: {height_px}px; border: 1px solid #e5e7eb; border-radius: 8px; }}
    </style>
  </head>
  <body>
    <div id=\"mynetwork\"></div>
    <script>
      const nodes = new vis.DataSet({nodes_json});
      const edges = new vis.DataSet({edges_json});
      const container = document.getElementById('mynetwork');
      const data = {{ nodes, edges }};
      const options = {{
        interaction: {{ hover: true, dragNodes: true, zoomView: true }},
        physics: {{ stabilization: true }},
        edges: {{ font: {{ align: 'middle' }}, smooth: {{ type: 'dynamic' }} }},
        nodes: {{ font: {{ color: '#111827' }} }}
      }};
      const network = new vis.Network(container, data, options);
      network.fit({{ animation: true }});
    </script>
  </body>
</html>
"""

