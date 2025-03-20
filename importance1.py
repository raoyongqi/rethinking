import pandas as pd
from sklearn.model_selection import KFold
from lofo import LOFOImportance, Dataset, plot_importance

# import data
train_df = pd.read_csv("data/selection.csv")
train_df = train_df.rename(columns={
    'hand_500m_china_03_08': 'hand',
    'hwsd_soil_clm_res_dom_mu': 'dom_mu',
    'hwsd_soil_clm_res_awt_soc': 'awt_soc'
})

if 'hwsd_soil_clm_res_pct_clay' in train_df.columns:
    train_df = train_df.rename(columns={'hwsd_soil_clm_res_pct_clay': 'pct_clay'})

sample_df = train_df.sample(frac=0.2, random_state=42)

print(sample_df.index)


cv = KFold(n_splits=4, shuffle=False) # Don't shuffle to keep the time split split validation

dataset = Dataset(df=sample_df, target="pathogen load", features=[col for col in train_df.columns if col != "pathogen load"])
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)

lofo_imp = LOFOImportance(dataset,model=model, cv=cv, scoring="neg_mean_squared_error")

importance_df = lofo_imp.get_importance()

importance_keep = importance_df.round(2)
importance_keep['range'] = importance_keep[['val_imp_0', 'val_imp_1', 'val_imp_2', 'val_imp_3']].apply(lambda x: f"（{x.min()}, {x.max()}）", axis=1)

importance_keep  = importance_keep[['feature','importance_mean', 'importance_std', 'range']]

importance_keep.to_excel("data/importance_result.xlsx", index=False)


import matplotlib.pyplot as plt
import pandas as pd
import warnings

def plot_importance(importance_df, figsize=(8, 8), kind="default", save_path=None):
    """Plot feature importance

    Parameters
    ----------
    importance_df : pandas dataframe
        Output dataframe from LOFO/FLOFO get_importance
    kind : string
        plot type can be default or box
    figsize : tuple
    save_path : string, optional
        Path to save the plot (e.g., 'importance_plot.svg' or 'importance_plot.pdf')
    """
    importance_df = importance_df.copy()
    importance_df["color"] = (importance_df["importance_mean"] > 0).map({True: 'g', False: 'r'})
    importance_df.sort_values("importance_mean", inplace=True)

    available_kinds = {"default", "box"}
    if kind not in available_kinds:
        warnings.warn(f"{kind} not in {available_kinds}. Setting to default")

    # Increase font size
    font_size = 30
    plt.rcParams.update({'font.size': font_size})  # Set default font size for the plot

    # Default bar plot
    if kind == "default":
        ax = importance_df.plot(x="feature", y="importance_mean", xerr="importance_std",
                                kind='barh', color=importance_df["color"], figsize=figsize)
        ax.set_xlabel('Importance', fontsize=font_size)
        ax.set_ylabel('Feature', fontsize=font_size)
        ax.set_title('')  # Remove the title

        # Increase the font size for ticks and labels
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.tick_params(axis='x', which='both', labelsize=font_size)  # Make sure x-axis ticks are also large
        ax.tick_params(axis='y', which='both', labelsize=font_size)  # Make sure y-axis ticks are also large

    # Box plot for "box" kind
    elif kind == "box":
        lofo_score_cols = [col for col in importance_df.columns if col.startswith("val_imp")]
        features = importance_df["feature"].values.tolist()
        importance_df.set_index("feature")[lofo_score_cols].T.boxplot(column=features, vert=False, figsize=figsize)
        plt.title('')  # Remove the title

        # Increase the font size for ticks and labels
        plt.tick_params(axis='both', which='major', labelsize=font_size)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
    plt.show()

# plot the means and standard deviations of the importances
plot_importance(importance_df, figsize=(20, 12),save_path='data/importance.png')