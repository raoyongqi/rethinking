import pandas as pd
import matplotlib.pyplot as plt

rmse_df = pd.read_csv("rmse_results.csv", index_col="ntree")
r2_df = pd.read_csv("r2_results.csv", index_col="ntree")

ntree_values = rmse_df.index
mtry_values = rmse_df.columns.astype(int)  

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for mtry in mtry_values:
    ax1.plot(ntree_values, rmse_df[str(mtry)], label=f'mtry={mtry}')
ax1.set_title('RMSE vs ntree and mtry')
ax1.set_xlabel('ntree')
ax1.set_ylabel('RMSE')
ax1.legend(title='mtry')

for mtry in mtry_values:
    ax2.plot(ntree_values, r2_df[str(mtry)], label=f'mtry={mtry}')
ax2.set_title('R² vs ntree and mtry')
ax2.set_xlabel('ntree')
ax2.set_ylabel('R²')
ax2.legend(title='mtry')

plt.tight_layout()
plt.savefig("ntreemtry.png")  

plt.show()
