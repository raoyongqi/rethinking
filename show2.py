import pandas as pd
import matplotlib.pyplot as plt
import scienceplots


results_df = pd.read_csv('data/model_results_1_to_1000_epochs.csv')

config = {
    "font.family": 'Arial',
    'font.size': 20,
  
}
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].plot(results_df['epochs'], results_df['train_r2'], label='Train R-Square', color='blue')
ax[0].plot(results_df['epochs'], results_df['test_r2'], label='Test R-Square', color='orange')

ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('R-Square')
ax[0].legend()

ax[1].plot(results_df['epochs'], results_df['train_loss'], label='Train Loss', color='blue')
ax[1].plot(results_df['epochs'], results_df['test_loss'], label='Test Loss', color='orange')

ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend()

plt.tight_layout()
plt.savefig("data/epoachs.png", dpi=300)

plt.show()
