import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
import pickle
from train_fingerprint_predictor import (
    ShapeFingerprintDataset, 
    FingerprintPredictor, 
    collate_shapes_and_fingerprints
)
from tqdm.auto import tqdm

def compute_metrics(true_fps, pred_fps):
    # Compute metrics per bit
    ap_scores = []
    auc_scores = []
    
    for i in range(true_fps.shape[1]):
        true_bit = true_fps[:, i]
        pred_bit = pred_fps[:, i]
        
        if len(np.unique(true_bit)) > 1:  # Only compute if there are both positive and negative samples
            ap_scores.append(average_precision_score(true_bit, pred_bit))
            auc_scores.append(roc_auc_score(true_bit, pred_bit))
    
    return {
        'mean_ap': np.mean(ap_scores),
        'std_ap': np.std(ap_scores),
        'mean_auc': np.mean(auc_scores),
        'std_auc': np.std(auc_scores)
    }

def evaluate_model(model, test_loader, device):
    model.eval()
    all_true_fps = []
    all_pred_fps = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            shapes = batch['shape'].to(device)
            true_fps = batch['fingerprint']
            
            pred_fps = model(shapes).cpu()
            
            all_true_fps.append(true_fps)
            all_pred_fps.append(pred_fps)
    
    all_true_fps = torch.cat(all_true_fps, dim=0).numpy()
    all_pred_fps = torch.cat(all_pred_fps, dim=0).numpy()
    
    metrics = compute_metrics(all_true_fps, all_pred_fps)
    return metrics

def main():
    # Load test data
    with open('/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/processed/validation/shape_fingerprint_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Create test dataset and loader
    test_dataset = ShapeFingerprintDataset(data)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_shapes_and_fingerprints)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FingerprintPredictor().to(device)
    model.load_state_dict(torch.load('best_fingerprint_predictor.pt', weights_only=True))
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device)
    
    print("\nEvaluation Results:")
    print(f"Mean Average Precision: {metrics['mean_ap']:.4f} ± {metrics['std_ap']:.4f}")
    print(f"Mean AUC-ROC: {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}")

if __name__ == "__main__":
    main() 