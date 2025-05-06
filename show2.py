import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# Load the CSV data
results_df = pd.read_csv('data/model_results_1_to_1000_epochs.csv')

# Set global font size
plt.rcParams.update({'font.size': 14})  # Increase font size globally

with plt.style.context('science'):
    # Create the figure and axes for the subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot on the left (Epoch vs R² scores)
    ax[0].plot(results_df['epochs'], results_df['train_r2'], label='Train R²', color='blue')
    ax[0].plot(results_df['epochs'], results_df['test_r2'], label='Test R²', color='orange')
    ax[0].set_title('Epoch vs R² Score')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('R² Score')
    ax[0].legend()

    # Plot on the right (Epoch vs Loss)
    ax[1].plot(results_df['epochs'], results_df['train_loss'], label='Train Loss', color='blue')
    ax[1].plot(results_df['epochs'], results_df['test_loss'], label='Test Loss', color='orange')
    ax[1].set_title('Epoch vs Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig("data/epoachs.png", dpi=300)

# Show the plots
plt.show()
