import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os.path


import mysrc.constants as cst


def plot_accuracy_vs_time(df, my_colors_, x_labels, filename=''):
    fig, axs = plt.subplots(df_.Crop.unique().shape[0], figsize=(8, 20))

    for i in range(df_.Crop.unique().shape[0]):
        df_i = df_.loc[df_.Crop == df_.Crop.unique()[i],].copy()
        for j, model_type in enumerate(df_i.Estimator.unique()):
            axs[i].plot(df_i.loc[df_i.Estimator == model_type, 'lead_time'].values,
                        df_i.loc[df_i.Estimator == model_type, 'rRMSE_p'].values,
                        color=my_colors_[j], label=model_type)
            axs[i].set_title(df_.Crop.unique()[i], fontweight="bold")
            axs[i].set_ylabel('rRMSE (%)')
            axs[i].set_ylim([9, 51])
            axs[i].set_yticks(range(10, 51, 10))
            axs[i].set_xticks(df_i.lead_time.unique())
            axs[i].set_xticklabels(x_labels)
            if i == (df_i.Estimator.unique().shape[0] - 1):
                axs[i].set_xlabel('Forcast date')
                axs[i].legend(loc="lower left", title="", frameon=False)
    plt.subplots_adjust(hspace=0.3)
    #plt.show()
    if filename != '':
        plt.savefig(filename, dpi=450)



# select the CNN file (produced by model_evaluation) to be compared to.
# model_evaluation saves in data, the file is moved to a new directory
CNNdir = cst.root_dir / f"data//Big run evaluation"
CNNfn = CNNdir / f"model_evaluation_1D_v_large_run_conv_filters_CNN.csv"
df_CNN_1D = pd.read_csv(CNNfn)
CNNfn = CNNdir / f"model_evaluation_2D_v10_big_run_avgPool_CNN.csv"
prefix_png_out = '1D_2D_MISO_v10_big_run_avgPool_CNN'
df_CNN = pd.read_csv(CNNfn)

target_var = 'Yield'
# -- Read in results
# ML and simple benchmarks
rdata_dir = Path(cst.root_dir, 'raw_data')
fn_benchmark = rdata_dir / r'all_model_output.csv'#best_ML_benchnarks.csv'
df_bench = pd.read_csv(fn_benchmark)
df_bench = df_bench.loc[:, ['lead_time', 'Crop', 'Estimator', 'rRMSE_p']].copy()
ML_selector = [False if x in ['PeakNDVI', 'Null_model'] else True for x in df_bench.Estimator]
df_bench.loc[ML_selector, 'Estimator'] = 'Machine Learning'
df_bench = df_bench.groupby(['lead_time', 'Crop', 'Estimator']).min().reset_index(level=[0, 1, 2], drop=False)
df_bench.loc[df_bench.Estimator == 'Null_model', 'Estimator'] = 'Null model'
df_bench.loc[df_bench.Estimator == 'PeakNDVI', 'Estimator'] = 'Peak NDVI'

# -- Plot Best 1D/2D CNN vs best benchmarks
x_tick_labels = ['Dec 1', 'Jan 1', 'Feb 1', 'Mar 1', 'Apr 1', 'May 1', 'Jun 1', 'Jul 1']
my_colors = ['#78b6fc', '#a9a9a9', '#ffc000' ]# '#034da2']

for crop_name in df_bench['Crop'].unique():
    fig, axs = plt.subplots(figsize=(12, 3), constrained_layout=True)
    #benchmark
    df_i = df_bench.loc[df_bench.Crop == crop_name].copy()
    mdl = 'Null model'
    axs.plot(df_i.loc[df_i.Estimator == mdl, 'lead_time'].values, df_i.loc[df_i.Estimator == mdl, 'rRMSE_p'].values,
             color='grey', linewidth=1, marker='o', label=mdl)
    mdl = 'Peak NDVI'
    axs.plot(df_i.loc[df_i.Estimator == mdl, 'lead_time'].values, df_i.loc[df_i.Estimator == mdl, 'rRMSE_p'].values,
             color='red', linewidth=1, marker='o', label=mdl)
    mdl = 'Machine Learning'
    axs.plot(df_i.loc[df_i.Estimator == mdl, 'lead_time'].values, df_i.loc[df_i.Estimator == mdl, 'rRMSE_p'].values,
             color='blue', linewidth=1, marker='o', label=mdl)
    # plot cnn results
    crop_ind = cst.crop_name_ind_dict[crop_name]
    df = df_CNN.loc[df_CNN.targetVar == target_var.lower()].copy()
    if crop_name in df.Crop.unique():
        dfc = df.loc[df.Crop == crop_name].copy()
        colors = ['lime','darkgreen','indigo','magenta']
        for j, model_type in enumerate(dfc.Estimator.unique()):
            axs.plot(dfc.loc[dfc.Estimator == model_type, 'lead_time'].values, dfc.loc[dfc.Estimator == model_type, 'rRMSE_p'].values, color=colors[j], linewidth=1, marker='o', label=model_type)

    df = df_CNN_1D.loc[df_CNN_1D.targetVar == target_var.lower()].copy()
    if crop_name in df.Crop.unique():
        dfc = df.loc[df.Crop == crop_name].copy()
        colors = ['indigo', 'magenta']
        for j, model_type in enumerate(dfc.Estimator.unique()):
            axs.plot(dfc.loc[dfc.Estimator == model_type, 'lead_time'].values,dfc.loc[dfc.Estimator == model_type, 'rRMSE_p'].values, color=colors[j], linewidth=1, marker='o', label=model_type)

    # tidy a bit and save
    axs.set_ylim(0, 50)
    axs.set_ylabel('rRMSEp (%)')
    axs.set_xticks(df_i.lead_time.unique())
    axs.set_xticklabels(x_tick_labels)
    axs.set_xlabel('Forecast time')
    # axs.set_title(crop_name + ', ' + y_var, fontsize=12)
    axs.set_title(crop_name, fontsize=12)
    # axs.set_title(axTilte,  fontsize=12)
    # fig.suptitle(crop_name + ', ' + y_var, fontsize=14, fontweight='bold')
    axs.legend(frameon=False, bbox_to_anchor=(1.04, 1), loc="upper left")
    #axs.legend(frameon=False, loc='upper right') #, ncol=len(axs.lines)
    fn = CNNdir / f'{prefix_png_out}_performances_{crop_name}.png'
    plt.savefig(fn, dpi=450)
    plt.close()
    print(crop_name + ' done')



