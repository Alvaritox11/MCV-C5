import os
import json
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm # Usamos tqdm.notebook para que la barra se vea bien en Jupyter
from torch.utils.data import DataLoader

# Importamos las utilidades directamente desde tu main.py
# (Asegúrate de que tu notebook está en la misma carpeta que main.py)
from main import evaluate, get_transforms, build_dataset, detection_collate_fn
from src.models.frnn_wrapper import FasterRCNNWrapper
from src.models.detr_wrapper import DetrWrapper
from src.models.rt_detr_wrapper import Rt_DetrWrapper

# Paleta de colores de tu compañero
LINE_COLORS = {
    "cyan": "#5BC0DE", 
    "tomato": "#FF7F6E", 
    "green": "#82B366", 
    "purple": "#D5AAFF", 
    "peach": "#FFD3B6"
}
SHORT_DASHES = (0, (2, 2))

import torch

def load_fused_lora_weights(model, checkpoint_path):
    """
    Carga un checkpoint de LoRA (PEFT) y fusiona matemáticamente las matrices 
    A y B dentro de los pesos base del modelo para inferencia estándar.
    """
    print(f"Loading and FUSING LoRA weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_raw = ckpt.get("state_dict", ckpt)
    
    # 1. Limpiar los prefijos que añade la librería PEFT de HuggingFace
    state = {k.replace("base_model.model.model.", "model."): v for k, v in state_raw.items()}
    
    # 2. Separar los pesos base (congelados) de los adaptadores LoRA
    base_state = {}
    for k, v in state.items():
        if "base_layer" in k:
            new_key = k.replace(".base_layer", "")
            base_state[new_key] = v
        elif "lora_A" not in k and "lora_B" not in k and "original_module" not in k and "modules_to_save" not in k:
            base_state[k] = v
            
    # 3. Fusionar matemáticamente LoRA en los pesos base: W = W_base + (lora_B @ lora_A)
    lora_keys = set()
    for k in state.keys():
        if "lora_A" in k:
            lora_keys.add(k.replace(".lora_A.default.weight", ""))
            
    for prefix in lora_keys:
        lora_A = state[f"{prefix}.lora_A.default.weight"]
        lora_B = state[f"{prefix}.lora_B.default.weight"]
        
        base_key = prefix + ".weight"
        if base_key in base_state:
            # Multiplicación de matrices y suma
            base_state[base_key] = base_state[base_key] + (lora_B @ lora_A)
        else:
            print(f"WARNING: base_key not found: {base_key}")

    # 4. Recuperar las cabezas de predicción (classifier y bbox_predictor)
    for k, v in state_raw.items():
        if "modules_to_save.default" in k and "original_module" not in k:
            new_key = k.replace("base_model.model.", "").replace("modules_to_save.default.", "")
            base_state[new_key] = v

    # 5. Cargar el diccionario fusionado en nuestro modelo estándar
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    print(f"  -> Fusion complete! Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    
    return model

def plot_threshold_tradeoff(df, output_dir, title="Confidence Threshold Trade-off"):
    """Plotea mAP vs Recall general."""
    plt.figure(figsize=(8, 5.5))
    
    plt.plot(df['threshold'], df['mAP_50_95'], label='Validation mAP @ 0.50:0.95', 
             linewidth=2.5, color=LINE_COLORS['cyan'], marker='o', markersize=8)
             
    plt.plot(df['threshold'], df['Recall_100'], label='Validation Recall (Max 100)', 
             linewidth=2.5, color=LINE_COLORS['tomato'], marker='s', markersize=8, linestyle=SHORT_DASHES)
    
    plt.xlabel("Confidence Threshold", fontsize=14, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=12)
    plt.xticks(df['threshold'], fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=12, frameon=True)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "threshold_tradeoff_plot_2.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()

def plot_threshold_sizes(df, output_dir, title="mAP by Object Size vs Threshold"):
    """Nuevo Plot: Plotea cómo afecta el threshold a los tamaños de objeto."""
    plt.figure(figsize=(8, 5.5))
    
    plt.plot(df['threshold'], df['mAP_Small'], label='Small Objects', 
             linewidth=2.3, color=LINE_COLORS['cyan'], marker='o')
    plt.plot(df['threshold'], df['mAP_Medium'], label='Medium Objects', 
             linewidth=2.3, color=LINE_COLORS['green'], marker='s')
    plt.plot(df['threshold'], df['mAP_Large'], label='Large Objects', 
             linewidth=2.3, color=LINE_COLORS['purple'], marker='^')
    
    plt.xlabel("Confidence Threshold", fontsize=14, fontweight="bold")
    plt.ylabel("mAP (%)", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=12)
    plt.xticks(df['threshold'], fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=12, frameon=True)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "threshold_sizes_plot_2.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()

def run_threshold_test(config_path, weights_path, output_folder_name="threshold_tests"):
    """
    Función principal para ejecutar el barrido de thresholds desde Jupyter.
    """
    # Crear carpeta de resultados
    output_dir = os.path.join("results", output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Threshold Search on {device} ---")

    # 1. Dataset & DataLoader (Usamos build_dataset de tu main.py actualizado)
    val_transforms = get_transforms("none", is_train=False) 
    val_dataset = build_dataset(root_dir=config["data_dir"], split="val", transforms=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 4), shuffle=False, collate_fn=detection_collate_fn)

    # 2. Model Initialization
    print(f"Initializing {config['model_type']}...")
    if config["model_type"] == "faster_rcnn":
        wrapper = FasterRCNNWrapper(device=device)
    elif config["model_type"] == "detr":
        wrapper = DetrWrapper(device=device, use_lora=config.get("use_lora", False), lora_r=config.get("lora_r", 8), lora_alpha=config.get("lora_alpha", 32))
    elif config["model_type"] == "rt_detr":
        wrapper = Rt_DetrWrapper(device=device, use_lora=config.get("use_lora", False), lora_r=config.get("lora_r", 8), lora_alpha=config.get("lora_alpha", 32))
    else:
        raise ValueError("Invalid model_type in config.")

    # 3. Load the Weights
    print(f"Loading weights from {weights_path}...")
    # if config.get("use_lora", False):
    #     # Si es un modelo LoRA, usamos la súper-función de tu compañero
    #     wrapper.model = load_fused_lora_weights(wrapper.model, weights_path)
    #     wrapper.model.to(device)
    # else:
        # Si es un modelo normal (Head-Only, Full FT), lo cargamos normal
    wrapper.model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        
    wrapper.model.eval()

    # 4. The Threshold Sweep
    thresholds_to_test = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    for t in thresholds_to_test:
        print(f"\n=== Evaluating with Confidence Threshold: {t} ===")
        config['confidence_threshold'] = t
        
        stats, map_car, map_ped, fps, val_loss, val_loss_ce, val_loss_bbox = evaluate(wrapper, val_loader, val_dataset, config)
        
        if stats is not None:
            # Stats array from COCOEvaluator:
            # 0: mAP_0.50_0.95_all | 1: mAP_0.50_all | 3: mAP_small | 4: mAP_medium | 5: mAP_large | 8: Recall_max100
            
            res_dict = {
                "threshold": t,
                "mAP_50_95": stats[0] * 100,
                "mAP_50": stats[1] * 100,
                "mAP_Small": stats[3] * 100 if stats[3] > -1 else 0.0,
                "mAP_Medium": stats[4] * 100 if stats[4] > -1 else 0.0,
                "mAP_Large": stats[5] * 100 if stats[5] > -1 else 0.0,
                "Recall_100": stats[8] * 100,
            }
            
            # Ajuste dinámico de clases según el dataset
            if val_dataset.dataset_type == "deart":
                res_dict["mAP_Person"] = map_ped * 100
            else:
                res_dict["mAP_Car"] = map_car * 100
                res_dict["mAP_Pedestrian"] = map_ped * 100
                
            print(f"Result -> mAP: {res_dict['mAP_50_95']:.2f}% | Recall: {res_dict['Recall_100']:.2f}% | Small: {res_dict['mAP_Small']:.2f}% | Large: {res_dict['mAP_Large']:.2f}%")
            results.append(res_dict)

    # 5. Save and Plot Results
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "threshold_results_DETR_2.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n--- Sweep Complete! Results saved to {csv_path} ---")
    
    # Generate the plots
    plot_threshold_tradeoff(df, output_dir, title=f"Threshold Trade-off ({config['model_type'].upper()})")
    plot_threshold_sizes(df, output_dir, title=f"mAP by Size vs Threshold ({config['model_type'].upper()})")

if __name__ == "__main__":
        # Reemplaza con tus rutas reales
    # mi_config = "configs/bests/frnn.json"
    # mis_pesos = "results/lora_sweep_jh41psgf_jh41psgf/best_model.pt"
    # # Esto creará una carpeta en results/mi_test_thresholds/ y guardará ahí el CSV y los gráficos
    # run_threshold_test(
    #     config_path=mi_config, 
    #     weights_path=mis_pesos, 
    #     output_folder_name="mi_test_thresholds" 
    # )

    mi_config = "configs/bests/detr.json"
    mis_pesos = "results/lora_sweep_w2vmrv4p_w2vmrv4p/best_model.pt"
    # Esto creará una carpeta en results/mi_test_thresholds/ y guardará ahí el CSV y los gráficos
    run_threshold_test(
        config_path=mi_config, 
        weights_path=mis_pesos, 
        output_folder_name="mi_test_thresholds" 
    )