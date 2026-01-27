import os
import glob

def print_latest_and_best(epoch_dir):
    # Print latest epoch file
    epoch_files = sorted(glob.glob(os.path.join(epoch_dir, 'epoch_*.txt')))
    if not epoch_files:
        print(f"No epoch files found in: {epoch_dir}")
    else:
        latest_file = epoch_files[-1]
        print(f"Latest epoch file: {latest_file}")
        with open(latest_file, 'r') as f:
            print(f.read())
    # Print best epoch info
    best_file = os.path.join(epoch_dir, 'best_epoch.txt')
    if os.path.exists(best_file):
        print(f"Best epoch info from: {best_file}")
        with open(best_file, 'r') as f:
            print(f.read())
    else:
        print(f"No best_epoch.txt found in: {epoch_dir}")

if __name__ == "__main__":
    # Overfit
    overfit_dir = os.path.join('checkpoints', 'debug_overfit_5songs_epochs')
    print("--- Overfit Loop ---")
    print_latest_and_best(overfit_dir)
    # Full train (stage 1 and 2)
    for stage in [1, 2]:
        full_dir = os.path.join('checkpoints', f'full_stage{stage}_epochs')
        print(f"--- Full Training Stage {stage} ---")
        print_latest_and_best(full_dir)