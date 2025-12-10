import torch
from monai.metrics import PSNRMetric, SSIMMetric, MAEMetric
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, device):
        self.device = device
        self.psnr_metric = PSNRMetric(max_val=1.0)
        self.ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
        self.mae_metric = MAEMetric()

    def update(self, outputs, targets):
        """Adds a batch of results to the running metrics."""
        self.psnr_metric(y_pred=outputs, y=targets)
        self.ssim_metric(y_pred=outputs, y=targets)
        self.mae_metric(y_pred=outputs, y=targets)

    def get_results(self):
        """Returns the aggregated results as a dictionary."""
        return {
            "PSNR": self.psnr_metric.aggregate().item(),
            "SSIM": self.ssim_metric.aggregate().item(),
            "MAE": self.mae_metric.aggregate().item()
        }

    def reset(self):
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self.mae_metric.reset()

def save_comparison_plot(inputs, targets, outputs, filename="result_preview.png"):
    """
    Helper to save a visual comparison of the middle slice.
    Expects tensors of shape [1, 1, D, H, W]
    """
    slice_idx = inputs.shape[4] // 2 
    
    img_in = inputs[0, 0, :, :, slice_idx].cpu().numpy()
    img_gt = targets[0, 0, :, :, slice_idx].cpu().numpy()
    img_out = outputs[0, 0, :, :, slice_idx].cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("Input (Contrast)"); plt.imshow(img_in, cmap="gray"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("Generated (Non-Contrast)"); plt.imshow(img_out, cmap="gray"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Ground Truth"); plt.imshow(img_gt, cmap="gray"); plt.axis('off')

    plt.savefig(filename)
    plt.close()