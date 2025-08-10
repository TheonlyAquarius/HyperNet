import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import glob
import matplotlib.pyplot as plt
from safetensors.torch import save_file, load_file
from .target_model import TargetModel
from .diffusion_model import AdvancedWeightSpaceDiffusion, flatten_state_dict, unflatten_to_state_dict, get_target_model_flat_dim

def evaluate_model_performance(model, test_loader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item() * data.size(0)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
    avg_loss = test_loss / total_samples
    accuracy = 100. * correct / total_samples
    return accuracy, avg_loss

def generate_checkpoints_with_diffusion(
    diffusion_model,
    initial_weights_flat,
    num_steps,
    target_model_reference_state_dict,
    device
):
    generated_weights_sequence_flat = []
    current_weights_flat = initial_weights_flat.to(device)
    diffusion_model.to(device)
    diffusion_model.eval()
    for t_idx in range(num_steps):
        timestep_tensor = torch.tensor([[float(t_idx)]], device=device)
        with torch.no_grad():
            predicted_next_weights_flat = diffusion_model(current_weights_flat.unsqueeze(0), timestep_tensor)
            predicted_next_weights_flat = predicted_next_weights_flat.squeeze(0)
        generated_weights_sequence_flat.append(predicted_next_weights_flat.cpu())
        current_weights_flat = predicted_next_weights_flat
        if (t_idx + 1) % 10 == 0 or (t_idx + 1) == num_steps:
            print(f"  Generated step {t_idx+1}/{num_steps}")
    return generated_weights_sequence_flat

def evaluate_diffusion_generated_checkpoints(
    diffusion_model_path,
    target_model_reference,
    checkpoints_OG,
    batch_size_eval=128,
    plot_results=True,
    download=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model_reference.to(device)
    reference_state_dict = target_model_reference.state_dict()
    target_flat_dim = get_target_model_flat_dim(reference_state_dict)
    time_emb_dim_diff = 256
    hidden_dim_diff = 1024
    num_layers_diff = 6
    num_heads_diff = 8
    dropout_diff = 0.1
    use_cross_attention = True
    use_adaptive_norm = True
    diffusion_model = AdvancedWeightSpaceDiffusion(
        target_model_flat_dim=target_flat_dim,
        time_emb_dim=time_emb_dim_diff,
        hidden_dim=hidden_dim_diff,
        num_layers=num_layers_diff,
        num_heads=num_heads_diff,
        dropout=dropout_diff,
        use_cross_attention=use_cross_attention,
        use_adaptive_norm=use_adaptive_norm
    )
    diffusion_model.load_state_dict(load_file(diffusion_model_path, device=device))
    diffusion_model.eval()
    new_random_model = TargetModel()
    initial_model_state_dict = new_random_model.state_dict()
    initial_weights_flat = flatten_state_dict(initial_model_state_dict).to(device)
    original_weight_files = sorted(
        glob.glob(os.path.join(checkpoints_OG, "weights_epoch_*.safetensors")),
        key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else -1
    )
    num_generation_steps = len(original_weight_files) - 1
    if num_generation_steps <= 0:
        print(f"Not enough weight files in {checkpoints_OG} to determine generation steps.")
        return
    generated_weights_flat_sequence = generate_checkpoints_with_diffusion(
        diffusion_model,
        initial_weights_flat,
        num_generation_steps,
        reference_state_dict,
        device
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=download, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    eval_model = TargetModel().to(device)
    accuracies_generated = []
    losses_generated = []
    eval_model.load_state_dict(unflatten_to_state_dict(initial_weights_flat.cpu(), reference_state_dict))
    acc, loss = evaluate_model_performance(eval_model, test_loader, device, criterion)
    accuracies_generated.append(acc)
    losses_generated.append(loss)
    print(f"Step 0 (Initial Random Weights): Accuracy = {acc:.2f}%, Avg Loss = {loss:.4f}")
    for i, flat_weights in enumerate(generated_weights_flat_sequence):
        generated_state_dict = unflatten_to_state_dict(flat_weights.cpu(), reference_state_dict)
        eval_model.load_state_dict(generated_state_dict)
        acc, loss = evaluate_model_performance(eval_model, test_loader, device, criterion)
        accuracies_generated.append(acc)
        losses_generated.append(loss)
        print(f"Generated Step {i+1}/{num_generation_steps}: Accuracy = {acc:.2f}%, Avg Loss = {loss:.4f}")
    if plot_results:
        plt.figure(figsize=(24, 10))
        plt.subplot(1, 2, 2)
        plt.plot(accuracies_generated, label="Diffusion Generated checkpoints", marker='o')
        plt.xlabel("Optimization Step / Epoch")
        plt.ylabel("Test Accuracy (%)")
        plt.title("Accuracy of Diffusion-Generated Weights")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 1)
        plt.plot(losses_generated, label="Diffusion Generated checkpoints", marker='o')
        plt.xlabel("Optimization Step / Epoch")
        plt.ylabel("Average Test Loss")
        plt.title("Loss of Diffusion-Generated Weights")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_save_path = "diffusion_evaluation_plot.png"
        plt.savefig(plot_save_path)
        print(f"Plot saved to {plot_save_path}")
    save_choice = input("Save the generated weights from this checkpoints? (yes/no): ").lower()
    if save_choice in ['yes', 'y']:
        save_dir = 'generalized_checkpoints_weights'
        os.makedirs(save_dir, exist_ok=True)
        save_file(initial_model_state_dict, os.path.join(save_dir, 'weights_step_0.safetensors'))
        for i, flat_weights in enumerate(generated_weights_flat_sequence):
            state_dict = unflatten_to_state_dict(flat_weights.cpu(), reference_state_dict)
            save_file(state_dict, os.path.join(save_dir, f'weights_step_{i+1}.safetensors'))
        print(f"Saved {len(generated_weights_flat_sequence) + 1} weight files to '{save_dir}'.")
    else:
        print("Generated weights were not saved.")
    print("Evaluation finished.")

# KEY
# download: dataset flag
# evaluate_model_performance: evaluator
# generate_checkpoints_with_diffusion: generator
# evaluate_diffusion_generated_checkpoints: evaluation pipeline
