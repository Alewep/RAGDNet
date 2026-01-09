import torch
import cv2
import matplotlib.pyplot as plt

from ragdnet.pipelines.factory import create_pipeline
from ragdnet.pipelines.image_to_graph.ragd.runner import RagdnetPipeline
from ragdnet.utils.converter import graph_to_img
from ragdnet.learning.datasets.graph_dataset import (
    nx_to_pointdata,
)


# --- CONFIG ------------------------------------------------------
CFG_PATH = r"packages\ragdnet\configs\image_to_graph\ragdnet_r75_115_k5.toml"
INPUT_PATH = r"...png"
CKPT_PATH = r"...ckpt"


# --- MODEL UTILS -------------------------------------------------
def get_model(pipe):
    if hasattr(pipe, "model") and pipe.model is not None:
        return pipe.model
    if hasattr(pipe, "runner") and hasattr(pipe.runner, "model"):
        return pipe.runner.model
    return None


def load_model(ckpt_path):
    from ragdnet.learning.models.gnn import GAT_L

    return GAT_L.load_from_checkpoint(ckpt_path)


# --- MAIN --------------------------------------------------------
def main():
    # 1) Charger le pipeline et l'image
    pipe:RagdnetPipeline = create_pipeline(CFG_PATH, RagdnetPipeline, "ragdnet.pipelines")
    pipe.labeler = None

    img = cv2.imread(INPUT_PATH)
    if img is None:
        raise FileNotFoundError(INPUT_PATH)

    # 2) Construire le graphe NetworkX
    graph = pipe.run(img, verbose=True)

    # 3) Convertir en PyG
    pyg_graph = nx_to_pointdata(graph)
    # 4) Charger le modèle
    model = get_model(pipe)
    if model is None:
        model = load_model(CKPT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 5) Inférence multi-classe
    with torch.no_grad():
        data = pyg_graph.to(device)
        out = model(data)  # (N, C)
        probs = torch.softmax(out, dim=-1)  # proba par classe
        y_pred = probs.argmax(dim=-1).cpu()  # classe prédite (entier)

    # 6) Affecter y_pred à chaque nœud
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]["y_pred"] = int(y_pred[i])
        graph.nodes[node]["y"] = int(y_pred[i])

    # 7) Visualisation
    img_out = graph_to_img(graph)
    plt.axis("off")
    plt.imshow(img_out)
    plt.show()


if __name__ == "__main__":
    main()
