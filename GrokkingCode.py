import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ModularAdditionDataset:
    def __init__(self, p=97, train_fraction=0.4, noise_level=0.0):
        self.p = p
        self.noise_level = noise_level

        # Generate all possible pairs
        all_pairs = [(i, j) for i in range(p) for j in range(p)]
        np.random.shuffle(all_pairs)

        # Split into train and test
        n_train = int(len(all_pairs) * train_fraction)
        self.train_pairs = all_pairs[:n_train]
        self.test_pairs = all_pairs[n_train:]

        # Create tensors
        self.X_train = torch.tensor(self.train_pairs, dtype=torch.long)
        self.y_train = torch.tensor([(x + y) % p for x, y in self.train_pairs], dtype=torch.long)

        self.X_test = torch.tensor(self.test_pairs, dtype=torch.long)
        self.y_test = torch.tensor([(x + y) % p for x, y in self.test_pairs], dtype=torch.long)

        # Store original clean labels
        self.y_train_clean = self.y_train.clone()

        # Generate noisy labels ONCE and store them for consistent evaluation
        self.y_train_noisy = self.y_train_clean.clone()
        if self.noise_level > 0:
            n_noisy = int(len(self.y_train_noisy) * self.noise_level)
            if n_noisy > 0:
                noisy_indices = torch.randperm(len(self.y_train_noisy))[:n_noisy]
                self.y_train_noisy[noisy_indices] = torch.randint(0, self.p, (n_noisy,))

    def get_noisy_train_batch(self, batch_size):
        """Get a batch of training data with consistent noise"""
        indices = torch.randperm(len(self.X_train))[:batch_size]
        X_batch = self.X_train[indices]
        y_batch = self.y_train_noisy[indices]  # Use pre-generated noisy labels
        return X_batch.to(device), y_batch.to(device)

    def get_full_train_set(self):
        """Get full training set with consistent noise"""
        # Return the pre-generated noisy labels (same every time)
        return self.X_train.to(device), self.y_train_noisy.to(device)

    def get_test_set(self):
        """Get clean test set (no noise)"""
        return self.X_test.to(device), self.y_test.to(device)

class GrokkingMLP(nn.Module):
    def __init__(self, p=97, hidden_size=512, embedding_dim=128):
        super().__init__()
        self.p = p

        # Embedding for inputs
        self.embedding = nn.Embedding(p, embedding_dim)

        # MLP layers - deeper network
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, p)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)  # No dropout for cleaner grokking

        # Initialize weights with smaller values for better grokking
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (batch_size, 2)
        x1_emb = self.embedding(x[:, 0])
        x2_emb = self.embedding(x[:, 1])
        x = torch.cat([x1_emb, x2_emb], dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x

def calculate_grokking_index(train_accs, test_accs, epoch_interval=1):
    """
    Calculate grokking index as area between accuracy curves.
    Integration starts at the EXACT epoch when training accuracy reaches at least 95%
    and ends when test accuracy is within 5% of training accuracy.
    """
    # Create array of exact epoch numbers
    epochs = np.arange(len(train_accs)) * epoch_interval

    # Find the EXACT epoch where train accuracy first reaches at least 95%
    start_idx = -1
    for i, acc in enumerate(train_accs):
        if acc >= 0.95:
            start_idx = i
            break

    if start_idx == -1:
        # Training never reached 95% - no grokking
        return 0.0

    # Find where test accuracy gets within 5% of training accuracy
    end_idx = len(test_accs)  # Default to end of training

    for i in range(start_idx, len(test_accs)):
        # Stop when test is within 5% of train
        if test_accs[i] >= train_accs[i] - 0.05:
            end_idx = i + 1  # Include the epoch where condition is met
            break
        # Also stop at exactly 50000 epochs
        if epochs[i] >= 50000:
            end_idx = i
            break

    # If no data between start and end, no grokking
    if end_idx <= start_idx:
        return 0.0

    # Slice arrays from exact start to exact end epoch
    train_accs_slice = np.array(train_accs[start_idx:end_idx])
    test_accs_slice = np.array(test_accs[start_idx:end_idx])
    epochs_slice = epochs[start_idx:end_idx]

    if len(epochs_slice) < 2:
        return 0.0

    # Calculate area between curves (train - test) using exact epoch-by-epoch data
    diff = train_accs_slice - test_accs_slice

    # Integrate using trapezoidal rule over exact epochs
    grokking_index = np.trapezoid(diff, epochs_slice)

    return grokking_index

def calculate_loss_index(train_losses, test_losses, train_accs, test_accs, epoch_interval=1):
    """
    Calculate loss index as area between loss curves.
    Uses the same integration bounds as the grokking index:
    - Starts when training accuracy reaches at least 95%
    - Ends when test accuracy is within 5% of training accuracy
    """
    # Create array of exact epoch numbers
    epochs = np.arange(len(train_losses)) * epoch_interval

    # Find the EXACT epoch where train accuracy first reaches at least 95%
    start_idx = -1
    for i, acc in enumerate(train_accs):
        if acc >= 0.95:
            start_idx = i
            break

    if start_idx == -1:
        # Training never reached 95% - no loss index
        return 0.0

    # Find where test accuracy gets within 5% of training accuracy
    end_idx = len(test_accs)  # Default to end of training

    for i in range(start_idx, len(test_accs)):
        # Stop when test is within 5% of train
        if test_accs[i] >= train_accs[i] - 0.05:
            end_idx = i + 1  # Include the epoch where condition is met
            break
        # Also stop at exactly 50000 epochs
        if epochs[i] >= 50000:
            end_idx = i
            break

    # If no data between start and end, no loss index
    if end_idx <= start_idx:
        return 0.0

    # Slice loss arrays from exact start to exact end epoch
    train_losses_slice = np.array(train_losses[start_idx:end_idx])
    test_losses_slice = np.array(test_losses[start_idx:end_idx])
    epochs_slice = epochs[start_idx:end_idx]

    if len(epochs_slice) < 2:
        return 0.0

    # Calculate area between curves (test - train) for losses
    # Note: For losses, we typically expect test > train during grokking
    diff = test_losses_slice - train_losses_slice

    # Integrate using trapezoidal rule over exact epochs
    loss_index = np.trapezoid(diff, epochs_slice)

    return loss_index

def train_model(noise_level, max_epochs=50000, batch_size=256):
    """Train a single model with given noise level"""

    # Create dataset with noise
    dataset = ModularAdditionDataset(p=97, train_fraction=0.4, noise_level=noise_level)

    # Create model
    model = GrokkingMLP(p=97, hidden_size=512, embedding_dim=128).to(device)

    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-3,  # Reduced from 1e-2 for more stability
        weight_decay=1.0  # Strong weight decay for grokking
    )

    # Add warmup for the first 100 epochs to prevent early instabilities
    def lr_lambda(epoch):
        warmup_epochs = 100
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        else:
            return 0.99995 ** (epoch - warmup_epochs)  # Then exponential decay

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss()

    # Tracking metrics - will store data for EVERY epoch
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    # Get test set (clean)
    X_test, y_test = dataset.get_test_set()

    # Get full training set for consistent evaluation
    X_train_full, y_train_noisy = dataset.get_full_train_set()

    # Training loop
    pbar = tqdm(range(max_epochs), desc=f"Noise {noise_level:.1%}")

    # Track milestones
    train_perfect_epoch = None
    test_perfect_epoch = None

    for epoch in pbar:
        # Training step
        model.train()

        # Always use mini-batches for stable training
        X_batch, y_batch = dataset.get_noisy_train_batch(min(batch_size, len(dataset.X_train)))

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Evaluation EVERY SINGLE EPOCH - both accuracy and loss
        model.eval()
        with torch.no_grad():
            # Training set evaluation
            train_outputs = model(X_train_full)
            train_loss = criterion(train_outputs, y_train_noisy).item()
            train_acc = (train_outputs.argmax(dim=1) == y_train_noisy).float().mean().item()

            # Test set evaluation (clean)
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_acc = (test_outputs.argmax(dim=1) == y_test).float().mean().item()

            # Record ALL metrics EVERY epoch
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            # Update progress bar every 100 epochs to avoid clutter
            if epoch % 100 == 0:
                pbar.set_postfix({
                    'ep': epoch,
                    'train': f'{train_acc:.3f}',
                    'test': f'{test_acc:.3f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            # EARLY STOPPING: only when train >= 80% AND test within 5% of train
            if train_acc >= 0.80 and test_acc >= train_acc - 0.05:
                print(f"\n  → Early stopping: Test within 5% of train (Train: {train_acc:.2%}, Test: {test_acc:.2%}) at epoch {epoch}")
                break

            # Also stop at max epochs (redundant but explicit)
            if epoch >= max_epochs - 1:
                print(f"\n  → Reached maximum epochs ({max_epochs})")
                break

    return train_losses, test_losses, train_accs, test_accs

def plot_training_curves(train_losses, test_losses, train_accs, test_accs, noise_level, run_idx):
    """Plot training curves for a single run - all data is epoch-by-epoch"""
    epochs = np.arange(len(train_accs))  # Every single epoch

    print(f"  Plotting {len(epochs)} data points (one per epoch)")
    print(f"  Loss data points: {len(train_losses)}, Accuracy data points: {len(train_accs)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot - data for EVERY epoch
    ax1.plot(epochs, train_losses, label='Train Loss', alpha=0.8, linewidth=1.5)
    ax1.plot(epochs, test_losses, label='Test Loss', alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Curves (Noise: {noise_level:.1%}, Run {run_idx+1}) - {len(train_losses)} points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Accuracy plot - EVERY epoch
    ax2.plot(epochs, train_accs, label='Train Accuracy', alpha=0.8, linewidth=1.5)
    ax2.plot(epochs, test_accs, label='Test Accuracy', alpha=0.8, linewidth=1.5)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Accuracy Curves (Noise: {noise_level:.1%}, Run {run_idx+1}) - {len(train_accs)} points')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Add vertical line where train acc reaches 95% (for grokking index)
    for i, acc in enumerate(train_accs):
        if acc >= 0.95:
            ax2.axvline(x=i, color='red', linestyle='--', alpha=0.5, linewidth=1.0, label='Train 95%')
            ax2.text(i, 0.5, f'Epoch {i}', rotation=90, verticalalignment='bottom')
            break

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Noise levels from 11% to 30% in 1% increments
    noise_levels = np.arange(0.0, 0.2, 0.01)
    n_runs = 5  # 5 random runs

    # Store results
    all_results = []

    print("=" * 50)
    print("GROKKING WITH DELAYED GENERALIZATION EXPERIMENT")
    print("=" * 50)
    print(f"Task: (x + y) mod 97")
    print(f"Noise levels: 0% to 15% in 0.5% increments ({len(noise_levels)} levels)")
    print(f"Runs per noise level: {n_runs} (random initializations)")
    print(f"Max epochs: 50,000")
    print(f"Tracking: EVERY EPOCH (not every 100)")
    print(f"Key settings:")
    print("  - Learning rate: 5e-3")
    print("  - Weight decay: 1.0")
    print("  - Hidden units: 512")
    print("  - Train/Test split: 40%/60%")
    print("  - Batch size: 256")
    print("  - Early stopping: When train ≥ 80% AND test ≥ train - 5%")
    print("  - Grokking Index: Area between accuracy curves (train - test)")
    print("  - Loss Index: Area between loss curves (test - train)")
    print("=" * 50)

    for noise_level in noise_levels:
        print(f"\n--- Noise Level: {noise_level:.1%} ---")

        for run_idx in range(n_runs):
            print(f"Run {run_idx + 1}/{n_runs}")

            # Train model
            train_losses, test_losses, train_accs, test_accs = train_model(
                noise_level=noise_level,
                max_epochs=50000,
                batch_size=256
            )

            # Calculate grokking index with exact epoch data
            grokking_index = calculate_grokking_index(train_accs, test_accs, epoch_interval=1)

            # Calculate loss index with the same bounds
            loss_index = calculate_loss_index(train_losses, test_losses, train_accs, test_accs, epoch_interval=1)

            # New: Calculate max accuracy gap
            max_acc_gap = max(np.array(train_accs) - np.array(test_accs)) if train_accs else 0

            # Debug output
            print(f"  Data points collected: {len(train_accs)} epochs")

            # Plot curves
            plot_training_curves(train_losses, test_losses, train_accs, test_accs,
                                 noise_level, run_idx)

            # Store results
            result = {
                'noise_level': noise_level,
                'run': run_idx + 1,
                'final_train_acc': train_accs[-1] if train_accs else 0,
                'final_test_acc': test_accs[-1] if test_accs else 0,
                'final_train_loss': train_losses[-1] if train_losses else float('inf'),
                'final_test_loss': test_losses[-1] if test_losses else float('inf'),
                'grokking_index': grokking_index,
                'loss_index': loss_index,
                'max_acc_gap': max_acc_gap,
                'epochs_trained': len(train_accs)
            }

            # Find EXACT epoch where train acc reaches 99% - LOGGING ONLY
            train_perfect_epoch = None
            for i, acc in enumerate(train_accs):
                if acc >= 0.99:
                    train_perfect_epoch = i
                    break
            result['train_perfect_epoch'] = train_perfect_epoch

            # Find EXACT epoch where test acc reaches 99% - LOGGING ONLY
            test_perfect_epoch = None
            for i, acc in enumerate(test_accs):
                if acc >= 0.99:
                    test_perfect_epoch = i
                    break
            result['test_perfect_epoch'] = test_perfect_epoch

            all_results.append(result)

            print(f"  Final Train Acc: {result['final_train_acc']:.2%}")
            print(f"  Final Test Acc: {result['final_test_acc']:.2%}")
            print(f"  Final Train Loss: {result['final_train_loss']:.4f}")
            print(f"  Final Test Loss: {result['final_test_loss']:.4f}")
            print(f"  Grokking Index: {result['grokking_index']:.2f}")
            print(f"  Loss Index: {result['loss_index']:.2f}")
            print(f"  Max Accuracy Gap: {result['max_acc_gap']:.2%}")
            print()

    # Convert results to DataFrame
    df = pd.DataFrame(all_results)

    # Calculate delay for runs that achieved 99% on both
    df['generalization_delay'] = df.apply(
        lambda row: row['test_perfect_epoch'] - row['train_perfect_epoch']
        if row['train_perfect_epoch'] is not None and row['test_perfect_epoch'] is not None else None,
        axis=1
    )

    # Summary statistics - Calculate averages for all 5 runs per noise level
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS - AVERAGED OVER 5 RUNS PER NOISE LEVEL")
    print("=" * 50)

    # Create comprehensive summary with means
    summary_means = df.groupby('noise_level').agg({
        'final_train_acc': 'mean',
        'final_test_acc': 'mean',
        'final_train_loss': 'mean',
        'final_test_loss': 'mean',
        'grokking_index': 'mean',
        'loss_index': 'mean',
        'max_acc_gap': 'mean',
        'epochs_trained': 'mean',
        'generalization_delay': 'mean'
    }).round(4)

    # Create comprehensive summary with standard deviations
    summary_stds = df.groupby('noise_level').agg({
        'final_train_acc': 'std',
        'final_test_acc': 'std',
        'final_train_loss': 'std',
        'final_test_loss': 'std',
        'grokking_index': 'std',
        'loss_index': 'std',
        'max_acc_gap': 'std',
        'epochs_trained': 'std',
        'generalization_delay': 'std'
    }).round(4)

    # Print detailed averages table
    print("\n=== MEAN VALUES (Averaged over 5 runs) ===")
    print(summary_means.to_string())

    print("\n=== STANDARD DEVIATIONS (Over 5 runs) ===")
    print(summary_stds.to_string())

    # Create a formatted summary table with mean ± std
    print("\n=== FORMATTED SUMMARY (Mean ± Std) ===")
    for noise in summary_means.index:
        print(f"\nNoise Level: {noise:.1%}")
        print("-" * 40)
        for col in ['final_train_acc', 'final_test_acc', 'grokking_index', 'loss_index', 'max_acc_gap']:
            mean_val = summary_means.loc[noise, col]
            std_val = summary_stds.loc[noise, col] if summary_stds.loc[noise, col] == summary_stds.loc[noise, col] else 0  # Handle NaN
            if col in ['final_train_acc', 'final_test_acc', 'max_acc_gap']:
                print(f"  {col:20s}: {mean_val:.2%} ± {std_val:.2%}")
            else:
                print(f"  {col:20s}: {mean_val:.2f} ± {std_val:.2f}")

        # Add epochs and delay info
        epochs_mean = summary_means.loc[noise, 'epochs_trained']
        epochs_std = summary_stds.loc[noise, 'epochs_trained'] if summary_stds.loc[noise, 'epochs_trained'] == summary_stds.loc[noise, 'epochs_trained'] else 0
        print(f"  {'epochs_trained':20s}: {epochs_mean:.0f} ± {epochs_std:.0f}")

        delay_mean = summary_means.loc[noise, 'generalization_delay']
        if delay_mean == delay_mean:  # Check for NaN
            delay_std = summary_stds.loc[noise, 'generalization_delay'] if summary_stds.loc[noise, 'generalization_delay'] == summary_stds.loc[noise, 'generalization_delay'] else 0
            print(f"  {'gen_delay':20s}: {delay_mean:.0f} ± {delay_std:.0f} epochs")

    # Plot summary with averaged values
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Grokking Analysis - Averaged Over 5 Runs Per Noise Level', fontsize=14, y=1.02)

    noise_levels_percent = df['noise_level'].unique() * 100

    # Mean accuracies
    ax = axes[0, 0]
    mean_train = df.groupby('noise_level')['final_train_acc'].mean()
    std_train = df.groupby('noise_level')['final_train_acc'].std()
    mean_test = df.groupby('noise_level')['final_test_acc'].mean()
    std_test = df.groupby('noise_level')['final_test_acc'].std()

    ax.errorbar(noise_levels_percent, mean_train, yerr=std_train,
                fmt='o-', label='Final Train Acc', alpha=0.7, capsize=5)
    ax.errorbar(noise_levels_percent, mean_test, yerr=std_test,
                fmt='s-', label='Final Test Acc', alpha=0.7, capsize=5)
    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Final Accuracy vs Noise Level (Mean ± Std)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Grokking index
    ax = axes[0, 1]
    mean_grokking = df.groupby('noise_level')['grokking_index'].mean()
    std_grokking = df.groupby('noise_level')['grokking_index'].std()
    ax.errorbar(noise_levels_percent, mean_grokking, yerr=std_grokking,
                 fmt='o-', alpha=0.7, capsize=5, color='green')
    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('Grokking Index')
    ax.set_title('Grokking Index vs Noise Level (Mean ± Std)')
    ax.grid(True, alpha=0.3)

    # Add text showing max grokking
    max_grok_idx = mean_grokking.idxmax()
    max_grok_val = mean_grokking.max()
    ax.annotate(f'Max: {max_grok_val:.1f} at {max_grok_idx:.1%}',
                xy=(max_grok_idx*100, max_grok_val),
                xytext=(max_grok_idx*100 + 2, max_grok_val),
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
                fontsize=9)

    # Loss index
    ax = axes[0, 2]
    mean_loss_index = df.groupby('noise_level')['loss_index'].mean()
    std_loss_index = df.groupby('noise_level')['loss_index'].std()
    ax.errorbar(noise_levels_percent, mean_loss_index, yerr=std_loss_index,
                 fmt='o-', alpha=0.7, capsize=5, color='orange')
    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('Loss Index')
    ax.set_title('Loss Index vs Noise Level (Mean ± Std)')
    ax.grid(True, alpha=0.3)

    # Add text showing max loss index
    max_loss_idx = mean_loss_index.idxmax()
    max_loss_val = mean_loss_index.max()
    ax.annotate(f'Max: {max_loss_val:.1f} at {max_loss_idx:.1%}',
                xy=(max_loss_idx*100, max_loss_val),
                xytext=(max_loss_idx*100 + 2, max_loss_val),
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
                fontsize=9)

    # Generalization delay
    ax = axes[1, 0]
    mean_delay = df.groupby('noise_level')['generalization_delay'].mean()
    std_delay = df.groupby('noise_level')['generalization_delay'].std()

    # Only plot where we have valid data (not NaN)
    valid_mask = ~mean_delay.isna()
    ax.errorbar(noise_levels_percent[valid_mask], mean_delay[valid_mask],
                yerr=std_delay[valid_mask],
                fmt='o-', color='purple', alpha=0.7, capsize=5)
    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('Generalization Delay (epochs)')
    ax.set_title('Average Delay Between 99% Train and 99% Test (Mean ± Std)')
    ax.grid(True, alpha=0.3)

    # Max Accuracy Gap
    ax = axes[1, 1]
    mean_max_acc_gap = df.groupby('noise_level')['max_acc_gap'].mean()
    std_max_acc_gap = df.groupby('noise_level')['max_acc_gap'].std()
    ax.errorbar(noise_levels_percent, mean_max_acc_gap, yerr=std_max_acc_gap,
                fmt='o-', alpha=0.7, capsize=5, color='red')
    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('Max Accuracy Gap')
    ax.set_title('Maximum Accuracy Gap vs Noise Level (Mean ± Std)')
    ax.grid(True, alpha=0.3)

    # Grokking vs Loss Index correlation
    ax = axes[1, 2]
    ax.scatter(df['grokking_index'], df['loss_index'], alpha=0.5)
    ax.set_xlabel('Grokking Index')
    ax.set_ylabel('Loss Index')
    ax.set_title('Grokking Index vs Loss Index')
    ax.grid(True, alpha=0.3)

    # Add correlation coefficient
    correlation = df[['grokking_index', 'loss_index']].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax.transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.show()

    # Save results
    df.to_csv('grokking_delayed_generalization_results.csv', index=False)
    print("\nResults saved to 'grokking_delayed_generalization_results.csv'")

    # Print grokking success rate
    print("\n" + "=" * 50)
    print("GROKKING SUCCESS ANALYSIS")
    print("=" * 50)

    for noise in df['noise_level'].unique():
        noise_df = df[df['noise_level'] == noise]
        n_total = len(noise_df)
        n_train_perfect = noise_df['train_perfect_epoch'].notna().sum()
        n_test_perfect = noise_df['test_perfect_epoch'].notna().sum()

        print(f"\nNoise {noise:.1%}:")
        print(f"  Runs reaching 99% train acc: {n_train_perfect}/{n_total}")
        print(f"  Runs reaching 99% test acc: {n_test_perfect}/{n_total}")

        if n_test_perfect > 0:
            avg_delay = noise_df['generalization_delay'].mean()
            if not np.isnan(avg_delay):
                print(f"  Average generalization delay: {avg_delay:.0f} epochs")

    # Print correlation analysis
    print("\n" + "=" * 50)
    print("CORRELATION ANALYSIS")
    print("=" * 50)
    print(f"Correlation between Grokking Index and Loss Index: {correlation:.3f}")

    # Additional correlations
    correlations = df[['grokking_index', 'loss_index', 'max_acc_gap', 'generalization_delay']].corr()
    print("\nFull Correlation Matrix:")
    print(correlations.round(3))

    # SPECIAL SUMMARY: Average Grokking Index and Loss Index for each noise level
    print("\n" + "=" * 50)
    print("AVERAGE GROKKING INDEX AND LOSS INDEX PER NOISE LEVEL")
    print("(Averaged across all 5 runs)")
    print("=" * 50)

    # Create a summary table
    index_summary = pd.DataFrame()
    for noise_level in sorted(df['noise_level'].unique()):
        noise_data = df[df['noise_level'] == noise_level]

        # Calculate averages for this noise level
        avg_grokking = noise_data['grokking_index'].mean()
        std_grokking = noise_data['grokking_index'].std()
        avg_loss = noise_data['loss_index'].mean()
        std_loss = noise_data['loss_index'].std()

        # Count successful runs
        n_runs = len(noise_data)
        n_grokked = (noise_data['grokking_index'] > 0).sum()

        index_summary = pd.concat([index_summary, pd.DataFrame({
            'Noise Level (%)': [f"{noise_level*100:.0f}%"],
            'Avg Grokking Index': [f"{avg_grokking:.2f}"],
            'Std Grokking Index': [f"{std_grokking:.2f}"],
            'Avg Loss Index': [f"{avg_loss:.2f}"],
            'Std Loss Index': [f"{std_loss:.2f}"],
            'Grokked Runs': [f"{n_grokked}/{n_runs}"]
        })], ignore_index=True)

    print("\n" + index_summary.to_string(index=False))

    # Find and report the optimal noise level
    optimal_grokking_noise = df.groupby('noise_level')['grokking_index'].mean().idxmax()
    optimal_grokking_value = df.groupby('noise_level')['grokking_index'].mean().max()
    optimal_loss_noise = df.groupby('noise_level')['loss_index'].mean().idxmax()
    optimal_loss_value = df.groupby('noise_level')['loss_index'].mean().max()

    print("\n" + "-" * 50)
    print(f"OPTIMAL NOISE LEVELS:")
    print(f"  Max Avg Grokking Index: {optimal_grokking_value:.2f} at {optimal_grokking_noise:.1%} noise")
    print(f"  Max Avg Loss Index: {optimal_loss_value:.2f} at {optimal_loss_noise:.1%} noise")
    print("-" * 50)

    # Create a final visualization comparing grokking and loss indices
    fig_final, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig_final.suptitle('Average Grokking and Loss Indices Across Noise Levels', fontsize=14)

    # Calculate means and stds for plotting
    noise_levels_unique = sorted(df['noise_level'].unique())
    grokking_means = [df[df['noise_level'] == nl]['grokking_index'].mean() for nl in noise_levels_unique]
    grokking_stds = [df[df['noise_level'] == nl]['grokking_index'].std() for nl in noise_levels_unique]
    loss_means = [df[df['noise_level'] == nl]['loss_index'].mean() for nl in noise_levels_unique]
    loss_stds = [df[df['noise_level'] == nl]['loss_index'].std() for nl in noise_levels_unique]

    # Plot Grokking Index
    ax1.errorbar(np.array(noise_levels_unique)*100, grokking_means, yerr=grokking_stds,
                 fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                 color='green', label='Grokking Index')
    ax1.set_xlabel('Noise Level (%)', fontsize=12)
    ax1.set_ylabel('Average Grokking Index', fontsize=12)
    ax1.set_title('Grokking Index vs Noise Level', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Mark the maximum
    max_idx = np.argmax(grokking_means)
    ax1.plot(noise_levels_unique[max_idx]*100, grokking_means[max_idx],
             'r*', markersize=15, label=f'Max: {grokking_means[max_idx]:.2f}')
    ax1.legend()

    # Plot Loss Index
    ax2.errorbar(np.array(noise_levels_unique)*100, loss_means, yerr=loss_stds,
                 fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                 color='orange', label='Loss Index')
    ax2.set_xlabel('Noise Level (%)', fontsize=12)
    ax2.set_ylabel('Average Loss Index', fontsize=12)
    ax2.set_title('Loss Index vs Noise Level', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Mark the maximum
    max_idx = np.argmax(loss_means)
    ax2.plot(noise_levels_unique[max_idx]*100, loss_means[max_idx],
             'r*', markersize=15, label=f'Max: {loss_means[max_idx]:.2f}')
    ax2.legend()

    plt.tight_layout()
    plt.show()
