import cv2
import networkx as nx
import numpy as np
from ragdnet.utils.color_mapper import map_id_to_color


def graph_to_img(
    graph: nx.Graph, background_color=(255, 255, 255), region_coloring: bool = False
):
    img = None
    H = W = None

    for i, attr in graph.nodes(data=True):
        coords = attr["src"]
        if coords.size == 0:
            continue

        if img is None:
            H, W = graph.graph["img_shape"]
            bg = np.array(background_color, dtype=np.uint8)
            img = np.tile(bg, (H, W, 1))

        rr, cc = coords[:, 0], coords[:, 1]
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        if not np.any(valid):
            continue
        rr, cc = rr[valid], cc[valid]
        if region_coloring:
            color = map_id_to_color(i + 1)
        else:
            color = map_id_to_color(attr["y"])
        color = np.array((color), dtype=np.uint8)

        img[rr, cc] = color

    return img


def _node_centroid_rc(coords: np.ndarray):
    if coords is None or coords.size == 0:
        return None
    r = float(coords[:, 0].mean())
    c = float(coords[:, 1].mean())
    return (r, c)


def _format_edge_label(val, max_len=32):
    if val is None:
        return ""
    try:
        arr = np.array(val)
        if arr.ndim == 0:
            s = str(arr.item())
        else:
            flat = arr.flatten()
            shown = flat[:4]
            s = (
                "["
                + ", ".join(f"{float(x):.2f}" for x in shown)
                + (", ..." if flat.size > 4 else "")
                + "]"
            )
    except Exception:
        s = str(val)

    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _draw_transparent_rect(img, pt1, pt2, color, alpha: float):
    if alpha <= 0:
        return img
    alpha = float(max(0.0, min(1.0, alpha)))

    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    return img


def render_region_graph_image(
    graph: nx.Graph,
    background_color=(255, 255, 255),
    region_coloring: bool = False,
    draw_edges: bool = True,
    edge_color=(255, 0, 0),
    edge_thickness: int = 1,
    arrow_tip_length: float = 0.05,
    edge_label_key: str = "feat",
    draw_edge_labels: bool = True,
    font_scale: float = 0.4,
    font_thickness: int = 1,
    label_bg_color=(255, 255, 255),
    label_bg_alpha: float = 0.5,
):
    img = None
    H = W = None

    # 1) Draw regions
    for i, attr in graph.nodes(data=True):
        coords = attr.get("src", None)
        if coords is None:
            continue
        coords = np.asarray(coords)
        if coords.size == 0:
            continue

        if img is None:
            H, W = graph.graph["img_shape"]
            bg = np.array(background_color, dtype=np.uint8)
            img = np.tile(bg, (H, W, 1))

        rr, cc = coords[:, 0], coords[:, 1]
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        if not np.any(valid):
            continue
        rr, cc = rr[valid], cc[valid]

        if region_coloring:
            color = map_id_to_color(i + 1)
        else:
            color = map_id_to_color(attr.get("y", 0))
        color = np.array(color, dtype=np.uint8)

        img[rr, cc] = color

    if img is None:
        return None

    # 2) Compute centroids
    centroids = {}
    for i, attr in graph.nodes(data=True):
        coords = attr.get("src", None)
        if coords is None:
            continue
        coords = np.asarray(coords)
        c = _node_centroid_rc(coords)
        if c is not None:
            centroids[i] = c

    is_directed = graph.is_directed()

    # 3) Draw edges (with arrowheads if directed)
    if draw_edges and centroids:
        for u, v, _ in graph.edges(data=True):
            if u not in centroids or v not in centroids:
                continue

            ru, cu = centroids[u]
            rv, cv = centroids[v]

            p1 = (int(round(cu)), int(round(ru)))  # (x=col, y=row)
            p2 = (int(round(cv)), int(round(rv)))

            if is_directed:
                cv2.arrowedLine(
                    img,
                    p1,
                    p2,
                    edge_color,
                    edge_thickness,
                    cv2.LINE_AA,
                    0,
                    arrow_tip_length,
                )
            else:
                cv2.line(
                    img,
                    p1,
                    p2,
                    edge_color,
                    thickness=edge_thickness,
                    lineType=cv2.LINE_AA,
                )

    # 4) Draw edge labels with transparent background
    if draw_edges and draw_edge_labels and centroids:
        font = cv2.FONT_HERSHEY_SIMPLEX

        for u, v, eattr in graph.edges(data=True):
            if u not in centroids or v not in centroids:
                continue

            label_val = eattr.get(edge_label_key, None)
            label = _format_edge_label(label_val)

            if label == "" and "edge_attr" in eattr:
                label = _format_edge_label(eattr["edge_attr"])
            if label == "":
                continue

            ru, cu = centroids[u]
            rv, cv = centroids[v]

            # place label slightly before the arrow head to reduce overlap
            t = 0.5 if not is_directed else 0.45
            rm = int(round(ru * (1 - t) + rv * t))
            cm = int(round(cu * (1 - t) + cv * t))

            (tw, th), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            pad = 2
            x0 = max(0, cm - tw // 2 - pad)
            y0 = max(0, rm - th // 2 - pad)
            x1 = min(W - 1, cm + tw // 2 + pad)
            y1 = min(H - 1, rm + th // 2 + pad + baseline)

            _draw_transparent_rect(
                img, (x0, y0), (x1, y1), label_bg_color, label_bg_alpha
            )

            org = (cm - tw // 2, rm + th // 2)
            cv2.putText(
                img,
                label,
                org,
                font,
                font_scale,
                (60, 60, 60),
                font_thickness,
                cv2.LINE_AA,
            )

    return img


def _node_centroid_rc(coords: np.ndarray):
    if coords is None or coords.size == 0:
        return None
    r = float(coords[:, 0].mean())
    c = float(coords[:, 1].mean())
    return (r, c)


def _format_scalar(val):
    try:
        return float(val)
    except Exception:
        return None


def _format_edge_label(val, max_len=32):
    if val is None:
        return ""
    try:
        arr = np.array(val)
        if arr.ndim == 0:
            s = str(arr.item())
        else:
            flat = arr.flatten()
            shown = flat[:4]
            s = (
                "["
                + ", ".join(f"{float(x):.2f}" for x in shown)
                + (", ..." if flat.size > 4 else "")
                + "]"
            )
    except Exception:
        s = str(val)

    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _draw_transparent_rect(img, pt1, pt2, color, alpha: float):
    if alpha <= 0:
        return img
    alpha = float(max(0.0, min(1.0, alpha)))
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    return img


def _extract_edge_attr(graph: nx.Graph, u, v):
    """
    Retourne un dict d'attributs pour (u,v).
    Supporte DiGraph et MultiDiGraph (prend la première arête).
    """
    data = graph.get_edge_data(u, v)
    if data is None:
        return None

    # MultiDiGraph: data est un dict de keys -> attrdict
    if isinstance(graph, (nx.MultiDiGraph, nx.MultiGraph)):
        if isinstance(data, dict) and len(data) > 0:
            first_key = next(iter(data.keys()))
            return data[first_key] if isinstance(data[first_key], dict) else {}
        return {}
    # DiGraph: data est directement un attrdict
    return data if isinstance(data, dict) else {}


def _get_label_from_attr(eattr, edge_label_key, fallback_key):
    if eattr is None:
        return None
    if edge_label_key in eattr:
        return eattr.get(edge_label_key)
    if fallback_key and fallback_key in eattr:
        return eattr.get(fallback_key)
    return None


def _merge_bidirectional_labels(val_uv, val_vu):
    """
    Fusion compacte des labels bidirectionnels.
    Spécial-cas pour scalaires numériques.
    """
    su = _format_scalar(val_uv)
    sv = _format_scalar(val_vu)

    if su is not None and sv is not None:
        if abs(su - sv) < 1e-6:
            return f"{su:.2f}"
        return f"{su:.2f}/{sv:.2f}"

    # fallback générique
    lu = _format_edge_label(val_uv)
    lv = _format_edge_label(val_vu)
    if lu and lv:
        if lu == lv:
            return lu
        # compact
        return f"{lu} | {lv}"
    return lu or lv or ""


def _draw_arrowhead(img, p_from, p_to, color, tip_length=0.18):
    """
    Dessine une tête de flèche au point p_to dans la direction p_from -> p_to.
    p_from, p_to en (x, y).
    tip_length est relatif à la longueur du segment.
    """
    x1, y1 = p_from
    x2, y2 = p_to
    vx = x2 - x1
    vy = y2 - y1
    length = (vx * vx + vy * vy) ** 0.5
    if length < 1e-6:
        return

    # longueur absolue de la tête
    arrow_len = max(6.0, length * float(tip_length))
    ux = vx / length
    uy = vy / length

    # perpendiculaire
    px = -uy
    py = ux

    base_x = x2 - ux * arrow_len
    base_y = y2 - uy * arrow_len

    base_w = arrow_len * 0.6

    p_tip = (int(round(x2)), int(round(y2)))
    p_left = (
        int(round(base_x + px * base_w / 2)),
        int(round(base_y + py * base_w / 2)),
    )
    p_right = (
        int(round(base_x - px * base_w / 2)),
        int(round(base_y - py * base_w / 2)),
    )

    pts = np.array([p_tip, p_left, p_right], dtype=np.int32)
    cv2.fillConvexPoly(img, pts, color)


def render_region_graph_image(
    graph: nx.Graph,
    background_color=(255, 255, 255),
    region_coloring: bool = True,
    draw_edges: bool = True,
    edge_color=(255, 0, 0),
    edge_thickness: int = 1,
    arrow_tip_length: float = 0.05,
    edge_label_key: str = "edge_attr",
    edge_label_fallback_key: str = "feat",
    draw_edge_labels: bool = True,
    font_scale: float = 0.4,
    font_thickness: int = 1,
    label_bg_color=(255, 255, 255),
    label_bg_alpha: float = 0.5,
):
    img = None
    H = W = None

    # 1) Draw regions
    for i, attr in graph.nodes(data=True):
        coords = attr.get("src", None)
        if coords is None:
            continue
        coords = np.asarray(coords)
        if coords.size == 0:
            continue

        if img is None:
            H, W = graph.graph["img_shape"]
            bg = np.array(background_color, dtype=np.uint8)
            img = np.tile(bg, (H, W, 1))

        rr, cc = coords[:, 0], coords[:, 1]
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        if not np.any(valid):
            continue
        rr, cc = rr[valid], cc[valid]

        if region_coloring:
            color = map_id_to_color(i + 1)
        else:
            color = map_id_to_color(attr.get("y", 0))
        color = np.array(color, dtype=np.uint8)

        img[rr, cc] = color

    if img is None:
        return None

    # 2) Compute centroids
    centroids = {}
    for i, attr in graph.nodes(data=True):
        coords = attr.get("src", None)
        if coords is None:
            continue
        coords = np.asarray(coords)
        c = _node_centroid_rc(coords)
        if c is not None:
            centroids[i] = c

    if not (draw_edges and centroids):
        return img

    is_directed = graph.is_directed()
    font = cv2.FONT_HERSHEY_SIMPLEX

    visited_pairs = set()

    # 3) Draw edges with merge for bidirectional pairs
    for u, v in graph.edges():
        if u not in centroids or v not in centroids:
            continue
        if u == v:
            continue

        pair = (u, v) if not is_directed else tuple(sorted((u, v)))
        if is_directed and pair in visited_pairs:
            continue
        if is_directed:
            visited_pairs.add(pair)

        ru, cu = centroids[u]
        rv, cv = centroids[v]

        p_u = (int(round(cu)), int(round(ru)))  # (x, y)
        p_v = (int(round(cv)), int(round(rv)))

        if not is_directed:
            # simple undirected
            cv2.line(
                img,
                p_u,
                p_v,
                edge_color,
                thickness=edge_thickness,
                lineType=cv2.LINE_AA,
            )
            continue

        has_uv = graph.has_edge(u, v)
        has_vu = graph.has_edge(v, u)

        # base line once
        cv2.line(
            img, p_u, p_v, edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA
        )

        # arrowheads
        if has_uv and has_vu:
            _draw_arrowhead(img, p_u, p_v, edge_color, tip_length=arrow_tip_length)
            _draw_arrowhead(img, p_v, p_u, edge_color, tip_length=arrow_tip_length)
        elif has_uv:
            _draw_arrowhead(img, p_u, p_v, edge_color, tip_length=arrow_tip_length)
        elif has_vu:
            _draw_arrowhead(img, p_v, p_u, edge_color, tip_length=arrow_tip_length)

        # 4) labels
        if draw_edge_labels:
            eattr_uv = _extract_edge_attr(graph, u, v) if has_uv else None
            eattr_vu = _extract_edge_attr(graph, v, u) if has_vu else None

            val_uv = _get_label_from_attr(
                eattr_uv, edge_label_key, edge_label_fallback_key
            )
            val_vu = _get_label_from_attr(
                eattr_vu, edge_label_key, edge_label_fallback_key
            )

            if has_uv and has_vu:
                label = _merge_bidirectional_labels(val_uv, val_vu)
            else:
                label = _format_edge_label(val_uv if has_uv else val_vu)

            if label:
                # place label at mid point
                rm = int(round((ru + rv) * 0.5))
                cm = int(round((cu + cv) * 0.5))

                (tw, th), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                pad = 2
                x0 = max(0, cm - tw // 2 - pad)
                y0 = max(0, rm - th // 2 - pad)
                x1 = min(W - 1, cm + tw // 2 + pad)
                y1 = min(H - 1, rm + th // 2 + pad + baseline)

                _draw_transparent_rect(
                    img, (x0, y0), (x1, y1), label_bg_color, label_bg_alpha
                )

                org = (cm - tw // 2, rm + th // 2)
                cv2.putText(
                    img,
                    label,
                    org,
                    font,
                    font_scale,
                    (60, 60, 60),
                    font_thickness,
                    cv2.LINE_AA,
                )

    return img
