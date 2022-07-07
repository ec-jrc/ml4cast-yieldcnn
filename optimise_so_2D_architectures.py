#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import optuna
from optuna.samplers import TPESampler
import joblib
import random
import wandb

random.seed(4)

#import tensorflow as tf
#tf.get_logger().setLevel('ERROR')

# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False

from deeplearning.architecture_cv import cv_Model
from deeplearning.architecture_complexity_2D import Archi_2DCNN #Archi_2DCNN_MISO, Archi_2DCNN_SISO
from outputfiles import plot as out_plot
from outputfiles import save as out_save
from evaluation import model_evaluation as mod_eval
from sits import readingsits2D, common_functions_1D2D
import mysrc.constants as cst
import datetime
import sits.data_generator_1D2D as data_generator
import global_variables


# global vars
version = '10_big_run_avgPool'
N_CHANNELS = 4  # -- NDVI, Rad, Rain, Temp
dict_train_params = {
    'optuna_metric': 'rmse', #'rmse' or 'r2'
    #'N_EPOCHS': 200, # 100, #70,
    #'BATCH_SIZE': 128, #128
    'N_TRIALS': 100,
    #'lr': 0.01, #0.001 is Adam default
    'beta_1': 0.9, #all defaults (they are not used now)
    'beta_2': 0.999,
    'decay':  0.01, # not used
    'l2_rate':   1.e-6 # This is strictly an archi parameters rather than a training one
}


# global vars - used in objective_2DCNN
dicthp = None
model_type = None
Xt = None
region_ohe = None
groups = None
data_augmentation = None
generator = None
y = None
crop_n = None
region_id = None
xlabels = None
ylabels = None
out_model = None
# to be used here and in architecture_cv


def optunaHyperSet2Test(Xd): #Xd id the time dimension of X
    x = {
        'nbunits_conv': {'low': 10, 'high': 30, 'step': 5}, # nbunits_conv_ = trial.suggest_int('nbunits_conv', 8, 48, step=4)
        'kernel_size': [3, 6, 9], #kernel_size_ = trial.suggest_int('kernel_size', 3, 6)
        'pool_size': {'low': 2, 'high': 3, 'step': 1}, #pool_size_ = trial.suggest_int('pool_size', 1, Xd // 3)
        'pyramid_bins':  {'low': 2, 'high': 3, 'step': 1},
        'dropout_rate': [0.01, 0.001], #dropout_rate_ = trial.suggest_float('dropout_rate', 0, 0.2, step=0.1)
        'learning_rate': [0.001, 0.01, 0.1],
        'fc_conf': [0, 1],
        # 'n_epochs': {'low': 30, 'high':  150, 'step': 20},
        'n_epochs': {'low': 100, 'high': 250, 'step': 50}, #{'low': 30, 'high': 90, 'step': 15}
        'batch_size': [64, 128]
    }
    return x


def main():
    starttime = datetime.datetime.now()
    # ---- Define parser
    parser = argparse.ArgumentParser(description='Optimise 2D CNN for yield and area forecasting')
    parser.add_argument('--normalisation', type=str, default='norm', choices=['norm', 'raw'],
                        help='Should input data be normalised histograms?')
    parser.add_argument('--model', type=str, default='2DCNN_MISO',
                        help='Model type: Single input single output (SISO) or Multiple inputs/Single output (MISO)')
    parser.add_argument('--target', type=str, default='yield', choices=['yield', 'area'], help='Target variable')
    parser.add_argument('--Xshift', dest='Xshift', action='store_true', default=False, help='Data aug, shiftX')
    parser.add_argument('--Xnoise', dest='Xnoise', action='store_true', default=False, help='Data aug, noiseX')
    parser.add_argument('--Ynoise', dest='Ynoise', action='store_true', default=False, help='Data aug, noiseY')
    parser.add_argument('--wandb', dest='wandb', action='store_true', default=False, help='Store results on wandb.io')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', default=False,
                        help='Overwrite existing results')
    args = parser.parse_args()

    # ---- Get parameters
    global model_type
    model_type = args.model
    if args.wandb:
        print('Wandb log requested')

    da_label = ''
    global data_augmentation
    if args.Xshift or args.Xnoise or args.Ynoise:
        data_augmentation = True
        if args.Xshift == True:
            da_label = da_label + 'Xshift'
        if args.Xnoise == True:
            da_label = da_label + '_Xnoise'
        if args.Ynoise == True:
            da_label = da_label + '_Ynoise'
    else:
        data_augmentation = False

    # ---- Define some paths to data
    fn_indata = cst.my_project.data_dir / f'{cst.target}_full_2d_dataset_raw.pickle'
    print("Input file: ", os.path.basename(str(fn_indata)))

    fn_asapID2AU = cst.root_dir / "raw_data" / "Algeria_REGION_id.csv"
    fn_stats90 = cst.root_dir / "raw_data" / "Algeria_stats90.csv"

    # for input_size in [32, 48, 64]:
    for input_size in [64,32]: #TODO now only one, put back [64, 32]
        # ---- Downloading (always not normalized)
        Xt_full, area_full, region_id_full, groups_full, yld_full = readingsits2D.data_reader(fn_indata)

        # M+ original resizing of Franz using tf.image.resize was bit odd as it uses bilinear interp (filling thus zeros)
        # resize if required (only resize to 32 possible)
        if input_size != 64:
            if input_size == 32:
                Xt_full = Xt_full.reshape(Xt_full.shape[0], -1, 2, Xt_full.shape[-2], Xt_full.shape[-1]).sum(2)
            else:
                print("Resizing request is not available")
                sys.exit()

        if args.normalisation == 'norm':
            max_per_image = np.max(Xt_full, axis=(1, 2), keepdims=True)
            Xt_full = Xt_full / max_per_image
        # M-

        # loop through all crops
        global crop_n
        for crop_n in [0,1,2]:  # range(y.shape[1]): TODO: now processing the two missing (0 - Barley, 1 - Durum, 2- Soft)
            # clean trial history for a new crop
            trial_history = []
            # keep only the data of the selected crop (the selected crop may not cover all the regions,
            # in the regions where it is not present, the yield was set to np nan when reading the data)
            # TODO: make sure there is no Nan
            yld_crop = yld_full[:, crop_n]
            subset_bool = ~np.isnan(yld_crop)
            yld = yld_crop[subset_bool]
            Xt_nozero = Xt_full[subset_bool, :, :, :]
            # make sure that we do not keep entries with 0 ton/ha yields,
            area = area_full[subset_bool, :]
            global region_id, groups
            region_id = region_id_full[subset_bool]
            groups = groups_full[subset_bool]
            # ---- Format target variable
            global y, xlabels, ylabels
            if args.target == 'yield':
                y = yld
                xlabels = 'Predictions (t/ha)'
                ylabels = 'Observations (t/ha)'
            elif args.target == 'area':
                y = area
                xlabels = 'Predictions (%)'
                ylabels = 'Observations (%)'

            # ---- Convert region to one hot
            global region_ohe
            region_ohe = common_functions_1D2D.add_one_hot(region_id)

            # loop by month
            for month in range(1, cst.n_month_analysis + 1): #range(1, cst.n_month_analysis + 1): #TODO put back all: range(1, cst.n_month_analysis + 1)
                # ---- output files and dirs
                dir_out = cst.my_project.params_dir
                dir_out.mkdir(parents=True, exist_ok=True)
                dir_res = dir_out / f'Archi_{str(model_type)}_{args.target}_{args.normalisation}_{input_size}_{da_label}'
                dir_res.mkdir(parents=True, exist_ok=True)
                global out_model
                #out_model = f'archi-{model_type}-{args.target}-{args.normalisation}.h5'
                out_model = f'{model_type}-{args.target}-{args.normalisation}_{input_size}_{da_label}.h5'
                # crop dirs
                dir_crop = dir_res / f'crop_{crop_n}' / f'v{version}'
                dir_crop.mkdir(parents=True, exist_ok=True)
                # month dirs

                global_variables.dir_tgt = dir_crop / f'month_{month}'
                global_variables.dir_tgt.mkdir(parents=True, exist_ok=True)

                if data_augmentation:
                    # Instantiate a data generator for this crop
                    global generator
                    generator = data_generator.DG(Xt_nozero, region_ohe, y, Xshift=args.Xshift, Xnoise=args.Xnoise,
                                                  Ynoise=args.Ynoise)

                if (len([x for x in global_variables.dir_tgt.glob('best_model')]) != 0) & (args.overwrite is False):
                    pass
                else:
                    # Clean up directory if incomplete run of if overwrite is True
                    out_save.rm_tree(global_variables.dir_tgt)
                    # data start in first dek of August (cst.first_month_in__raw_data), index 0
                    # the model uses data from first dek of September (to account for precipitation, field preparation),
                    # cst.first_month_input_local_year, =1, 1*3, index 3
                    # first forecast (month 1) is using up to end of Nov, index 11
                    first = (cst.first_month_input_local_year) * 3
                    last = (cst.first_month_analysis_local_year + month - 1) * 3  # this is 12
                    global Xt
                    Xt = Xt_nozero[:, :, first:last, :]  # this takes 9 elements, from 3 to 11 included
                    # Define and save hyper domain to test
                    global dicthp
                    dicthp = optunaHyperSet2Test(Xt.shape[1])
                    fn_hp = global_variables.dir_tgt / f'AAA_model_hp_tested_{version}.txt'
                    with open(fn_hp, 'w') as f:
                        f.write('hyper space tested\n')
                        for key in dicthp.keys():
                            f.write("%s,%s\n" % (key, dicthp[key]))  #
                        f.write('train parameters\n')  # dict_train_params
                        for key in dict_train_params.keys():
                            f.write("%s,%s\n" % (key, dict_train_params[key]))
                    print('------------------------------------------------')
                    print('------------------------------------------------')
                    print(f"")
                    print(f'=> archi: {model_type} - normalisation: {args.normalisation} - target:'
                          f' {args.target} - crop: {crop_n} - month: {month}')
                    print(f'Training data have shape: {Xt.shape}')
                    if dict_train_params['optuna_metric'] == 'rmse':
                        dirct = 'minimize'
                    elif dict_train_params['optuna_metric'] == 'r2':
                        dirct = 'maximize'
                    study = optuna.create_study(direction=dirct,
                                                sampler=TPESampler(seed=10),
                                                pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=8)
                                                )

                    # Force the sampler to sample at previously best model configuration
                    if len(trial_history) > 0:
                        for best_previous_trial in trial_history:
                            study.enqueue_trial(best_previous_trial)

                    study.optimize(objective_2DCNN, n_trials=dict_train_params['N_TRIALS'])

                    trial = study.best_trial
                    print('------------------------------------------------')
                    print('--------------- Optimisation results -----------')
                    print('------------------------------------------------')
                    print("Number of finished trials: ", len(study.trials))
                    print(f"\n           Best trial ({trial.number})        \n")
                    print(dict_train_params['optuna_metric']+": ", trial.value)
                    print("Params: ")
                    for key, value in trial.params.items():
                        print("{}: {}".format(key, value))
                    trial_history.append(trial.params)

                    joblib.dump(study, os.path.join(global_variables.dir_tgt, f'study_{crop_n}_{model_type}.dump'))
                    # dumped_study = joblib.load(os.path.join(cst.my_project.meta_dir, 'study_in_memory_storage.dump'))
                    # dumped_study.trials_dataframe()
                    df = study.trials_dataframe().to_csv(os.path.join(global_variables.dir_tgt, f'study_{crop_n}_{model_type}.csv'))
                    # fig = optuna.visualization.plot_slice(study)
                    print('------------------------------------------------')

                    out_save.save_best_model(global_variables.dir_tgt, f'trial_{trial.number}')

                    # Flexible integration for any Python script
                    if args.wandb:
                        run_wandb(args, month, input_size, trial, da_label, fn_asapID2AU, fn_stats90)
    print('Time for this run:')
    print(datetime.datetime.now() - starttime)
    # dir_res
    fn_hp = dir_res / f'AAA_executition_time.txt'
    with open(fn_hp, 'w') as f:
        f.write('Time for this run:\n')
        f.write(str(datetime.datetime.now() - starttime))


def objective_2DCNN(trial):
    #TODo arrived here on 2021-11-29
    global_variables.trial_number = trial.number
    Xt_ = Xt
    # Input dimension
    Yd = Xt_.shape[1]    #64 or 32
    Xd = Xt_.shape[2]    # 9, 12, 15, .., 30
    # Suggest values of the hyperparameters using a trial object.
    nbunits_conv_ = trial.suggest_int('nbunits_conv', dicthp['nbunits_conv']['low'], dicthp['nbunits_conv']['high'],
                                      step=dicthp['nbunits_conv']['step'])
    kernel_size_ = trial.suggest_categorical('kernel_size', dicthp['kernel_size'])
    pool_size_ = trial.suggest_int('pool_size', dicthp['pool_size']['low'], dicthp['pool_size']['high'],
                                   step=dicthp['pool_size'][
                                       'step'])  # should we fix it at 3, monthly pooling (with max)
    strides_ = pool_size_
    dropout_rate_ = trial.suggest_categorical('dropout_rate', dicthp['dropout_rate'])
    learning_rate_ = trial.suggest_categorical('learning_rate', dicthp['learning_rate'])
    n_epochs_ = trial.suggest_int('n_epochs', dicthp['n_epochs']['low'], dicthp['n_epochs']['high'],
                                  step=dicthp['n_epochs']['step'])  # 210
    batch_size_ = trial.suggest_categorical('batch_size', dicthp['batch_size'])

    fc_conf = trial.suggest_categorical('fc_conf', dicthp['fc_conf'])
    pyramid_bins_ = trial.suggest_int('pyramid_bins', dicthp['pyramid_bins']['low'], dicthp['pyramid_bins']['high'],
                                  step=dicthp['pyramid_bins']['step'])
    if False:
        nbunits_conv_ = 15
        kernel_size_ = 3
        pool_size_ = 3
        strides_ = pool_size_
        dropout_rate_ = 0
        learning_rate_ = 0.02
        n_epochs_ = 100
        batch_size_ = 128
        fc_conf = 0
        pyramid_bins_ = 2
    #old way:
    # n filters in the convolutions & n units in the dense layer after Xv (region Id OHE)
    #nbunits_conv_ = trial.suggest_int('nbunits_conv', 8, 48, step=4)
    # size of convolutions kernels
    #kernel_size_ = trial.suggest_int('kernel_size', 3, 6)
    # > using padding "same" the x and y dimension are not changed (64 or 32, n_month*3)

    # size of avg pooling layer between the two convolutions (now using padding "valid", also because using avg)
    # set max to Xd/3 to avoid over downsampling
    #pool_size_ = trial.suggest_int('pool_size', 1, Xd // 3)               # old Franz comment POOL SIZE Y, and let strides = pool size (//2 on time axis)
    # strides of avg pooling layer between the two convolutions
    #strides_ = pool_size_ #trial.suggest_int('strides', 1, pool_size_)                         # old Franz comment: MAKE IT POOL SIZE

    # here we change the dimension of the image, pyramids shall adapt to avoid asking more pyramid than image size
    # new dimensions:
    # with padding "valid"
    #output_shape = math.floor((input_shape - pool_size) / strides) + 1(when input_shape >= pool_size)
    Xdp = (Xd - pool_size_) // strides_ + 1
    Ydp = (Yd - pool_size_) // strides_ + 1
    print('Dims after 2D pooling', Ydp, Xdp)
    # pyramid bins (make sure that we do not ask more bins than dimension)
    max_pyramid_bins = np.min([pyramid_bins_, np.min([Xdp, Ydp])])
    #pyramid_bins_ = trial.suggest_int('pyramid_bin', 1, 4)
    pyramid_bins_ = [[k,k] for k in np.arange(1, max_pyramid_bins+1)]
    # drop ou for conv1, conv2 and final dens layers
    #dropout_rate_ = trial.suggest_float('dropout_rate', 0, 0.2, step=0.1)
    # number of final dense layers before output (0,1,2)
    #nb_fc_ = trial.suggest_categorical('nb_fc', [0, 1, 2])
    # number n of units in the first final dense layer before output, second layer will have n/2, third n/4
    #nunits_fc_ = trial.suggest_int('funits_fc', 16, 64, step=8) #the additional fc layer will have n, n/2, n/4 units
    #activation_ = trial.suggest_categorical('activation', ['relu', 'sigmoid'])

    if fc_conf == 0:
        nb_fc_ = 0
        nunits_fc_ = 0
    elif fc_conf == 1:
        nb_fc_ = 1
        nunits_fc_ = 16
    elif fc_conf == 2:
        nb_fc_ = 2 #nb_fc_ = 1
        nunits_fc_ = 24

    if model_type == '2DCNN_SISO':
        model = Archi_2DCNN('SISO',Xt_,
                                 nbunits_conv=nbunits_conv_,
                                 kernel_size=kernel_size_,
                                 strides=strides_,
                                 pool_size=pool_size_,
                                 pyramid_bins=pyramid_bins_,
                                 dropout_rate=dropout_rate_,
                                 nb_fc=nb_fc_,
                                 nunits_fc=nunits_fc_,
                                 l2_rate=dict_train_params['l2_rate'],
                                 activation='sigmoid',
                                 verbose=False)

    elif model_type == '2DCNN_MISO':
        model = Archi_2DCNN('MISO',Xt_,
                                 Xv=region_ohe,
                                 nbunits_conv=nbunits_conv_,
                                 kernel_size=kernel_size_,
                                 strides=strides_,
                                 pool_size=pool_size_,
                                 pyramid_bins=pyramid_bins_,
                                 dropout_rate=dropout_rate_,
                                 nb_fc=nb_fc_,
                                 nunits_fc=nunits_fc_,
                                 activation='sigmoid',
                                 l2_rate=dict_train_params['l2_rate'],
                                 verbose=False)
    print('Model hypars being tested')
    n_dense_before_output = (len(model.layers) - 1 - 14 - 1) / 2
    hp_dic = {'lr': learning_rate_,
              'cn_fc4Xv_units': model.layers[1].get_config()['filters'],
              'cn kernel_size': model.layers[1].get_config()['kernel_size'],
              #'cn strides (fixed)': str(model.layers[1].get_config()['strides']),
              'cn drop out rate': model.layers[4].get_config()['rate'],
              'AveragePooling2D pool_size': model.layers[5].get_config()['pool_size'],
              'AveragePooling2D strides': model.layers[5].get_config()['strides'],
              'SpatialPyramidPooling2D bins': model.layers[10].get_config()['bins'],
              'n FC layers before output (nb_fc)': int(n_dense_before_output),
              'n_epochs': n_epochs_,
              'batch_size_': batch_size_
              }
    # for i in range(int(n_dense_before_output)):
    #     hp_dic[str(i) + ' ' + 'fc_units'] = str(model.layers[15 + i * 2].get_config()['units'])
    #     hp_dic[str(i) + ' ' + 'drop out rate'] = str(model.layers[16 + i * 2].get_config()['rate'])
    # print(hp_dic.values())
    dorWithoutDot = str(hp_dic["cn drop out rate"]).replace('.', '-')
    hpsString = f'cnu{hp_dic["cn_fc4Xv_units"]}k{hp_dic["cn kernel_size"][0]}d{dorWithoutDot}' \
                f'p2Dsz_st{hp_dic["AveragePooling2D pool_size"][0]}_{hp_dic["AveragePooling2D strides"][0]}pyr{max(hp_dic["SpatialPyramidPooling2D bins"])[0]}'
    # hpsString = '_cn'+hp_dic['cn_fc4Xv_units']+'krnl'+hp_dic['cn kernel_size'][0]+'dor'+hp_dic['cn drop out rate']+'p2Dsz'+hp_dic['AveragePooling2D pool_size'][0] + \
    #     'p2Dstr'+hp_dic['AveragePooling2D strides'][0]+'pyr'+max(hp_dic['SpatialPyramidPooling2D bins'])[0]
    for i in range(int(n_dense_before_output)):
        if i == 0:
            hpsString = hpsString + 'dns'+ str(model.layers[15 + i * 2].get_config()['units'])
        else:
            hpsString = hpsString +'-'+ str(model.layers[15 + i * 2].get_config()['units'])
    print(hpsString)
    # Define output filenames
    fn_fig_val = global_variables.dir_tgt / f'trial_{trial.number}_{hpsString}_val.png'
    fn_fig_test = global_variables.dir_tgt / f'trial_{trial.number}_{hpsString}_test.png'
    fn_cv_test = global_variables.dir_tgt / f'trial_{trial.number}_{hpsString}_test.csv'
    fn_report = global_variables.dir_tgt / f'AAA_report_{version}.csv'
    out_model_file = global_variables.dir_tgt / f'{out_model.split(".h5")[0]}_{crop_n}.h5'

    #mses_val, r2s_val, mses_test, r2s_test = [], [], [], []
    #df_val, df_test, df_details = None, None, None
    #cv_i = 0
    #global_variables.init_weights = None

    rmses_train, r2s_train, rmses_val, r2s_val, rmses_test, r2s_test = [], [], [], [], [], []
    df_train, df_val, df_test, df_details, df_bestEpoch = None, None, None, None, None
    global_variables.init_weights = None

    sampleTerciles = True
    nPerTercile = 2

    global_variables.outer_test_loop = 0
    for test_i in np.unique(groups):
        global_variables.test_group = test_i
        global_variables.inner_cv_loop = 0
        # once the test is excluded, all the others are train and val
        train_val_i = [x for x in np.unique(groups) if x != test_i]
        subset_bool = groups == test_i
        Xt_test, Xv_test, y_test = Xt_[subset_bool, :, :, :], region_ohe[subset_bool, :], y[subset_bool]
        # a validation loop on all 16 years of val is too long. We reduce to nPerTercile*3,
        # taking 2 from each tercile of the yield data points
        # Here we assign a tercile to each year. As I have several admin units, I have first to compute avg yield by year
        if sampleTerciles:
            subset_bool = groups > 0 # take all
            Xt_0, Xv_0, y_0 = Xt_[subset_bool, :, :, :], region_ohe[subset_bool, :], y[subset_bool]
            #Xt_0, Xv_0, y_0 = readingsits1D.subset_data(Xt, region_ohe, y, groups > 0)  # take all
            df = pd.DataFrame({'group': groups, 'y': y_0})
            df = df[df['group'] != test_i]
            df_avg_by_year = df.groupby('group', as_index=False)['y'].mean()
            q33 = df_avg_by_year['y'].quantile(0.33)
            q66 = df_avg_by_year['y'].quantile(0.66)
            ter1_groups = df_avg_by_year[df_avg_by_year['y'] < q33]['group'].values
            ter2_groups = df_avg_by_year[(df_avg_by_year['y'] >= q33) & (df_avg_by_year['y'] < q66)]['group'].values
            ter3_groups = df_avg_by_year[df_avg_by_year['y'] >= q66]['group'].values
            vals_i = random.sample(ter1_groups.tolist(), nPerTercile)
            vals_i.extend(random.sample(ter2_groups.tolist(), nPerTercile))
            vals_i.extend(random.sample(ter3_groups.tolist(), nPerTercile))
        else:
            vals_i = train_val_i

        for val_i in vals_i:  # leave one out for hyper setting (in a way)
            global_variables.val_group = val_i
            # once the val is left out, all the others are train
            train_i = [x for x in train_val_i if x != val_i]
            subset_bool = [x in train_i for x in groups]
            Xt_train, Xv_train, y_train = Xt_[subset_bool, :, :, :], region_ohe[subset_bool, :], y[subset_bool]

            #*************************************

            # training data augmentation
            if data_augmentation:
                Xt_train, Xv_train, y_train = generator.generate(Xt_train.shape[2], subset_bool)

            subset_bool = groups == val_i
            Xt_val, Xv_val, y_val = Xt_[subset_bool, :, :, :], region_ohe[subset_bool, :], y[subset_bool]
            # If images are already normalised per region, the following has no effect
            # if not this is a minmax scaling based on the training set.
            # WARNING: if data are normalized by region (and not by image), the following normalisation would have an effect
            min_per_t, max_per_t = readingsits2D.computingMinMax(Xt_train, per=0)
            # Normalise training set
            Xt_train = readingsits2D.normalizingData(Xt_train, min_per_t, max_per_t)
            # print(f'Shape training data: {Xt_train.shape}')
            # Normalise validation set
            Xt_val = readingsits2D.normalizingData(Xt_val, min_per_t, max_per_t)

            # Normalise ys
            transformer_y = MinMaxScaler().fit(y_train.reshape(-1, 1))
            ys_train = transformer_y.transform(y_train.reshape(-1, 1))
            ys_val = transformer_y.transform(y_val.reshape(-1, 1))

            # Compile and fit
            if model_type == '2DCNN_SISO':
                model, y_val_preds, bestEpoch = cv_Model(model, {'ts_input': Xt_train}, ys_train,
                                              {'ts_input': Xt_val}, ys_val,
                                              out_model_file, n_epochs=n_epochs_,
                                              batch_size=batch_size_,
                                              learning_rate=learning_rate_, beta_1=dict_train_params['beta_1'],
                                              beta_2=dict_train_params['beta_2'], decay=dict_train_params['decay'])
                X_test = {'ts_input': Xt_test}
                y_train_preds = model.predict(x={'ts_input': Xt_train})
            elif model_type == '2DCNN_MISO':
                model, y_val_preds, bestEpoch = cv_Model(model, {'ts_input': Xt_train, 'v_input': Xv_train}, ys_train,
                                              {'ts_input': Xt_val, 'v_input': Xv_val}, ys_val,
                                              out_model_file, n_epochs=n_epochs_,
                                              batch_size=batch_size_,
                                              learning_rate=learning_rate_, beta_1=dict_train_params['beta_1'],
                                              beta_2=dict_train_params['beta_2'], decay=dict_train_params['decay'])
                X_test = {'ts_input': Xt_test, 'v_input': Xv_test}
                y_train_preds = model.predict(x={'ts_input': Xt_train, 'v_input': Xv_train})

            y_val_preds = transformer_y.inverse_transform(y_val_preds)
            out_val = np.concatenate([y_val.reshape(-1, 1), y_val_preds], axis=1)

            y_train_preds = transformer_y.inverse_transform(y_train_preds)
            out_train = np.concatenate([y_train.reshape(-1, 1), y_train_preds], axis=1)

            if df_val is None:
                df_val = out_val
            else:
                df_val = np.concatenate([df_val, out_val], axis=0)
            if df_train is None:
                df_train = out_train
            else:
                df_train = np.concatenate([df_train, out_train], axis=0)
            if df_bestEpoch is None:
                df_bestEpoch = np.array(bestEpoch)
            else:
                df_bestEpoch = np.append(df_bestEpoch, bestEpoch)

            # It happens that the trial results in  y_val_preds being nan because model fit failed with given optuna params and data
            # To avoid rasin nan errors in computation of stats below we handle this here
            if np.isnan(y_val_preds).any():
                rmses_val.append(np.nan)
                r2s_val.append(np.nan)
                rmses_train.append(np.nan)
                r2s_train.append(np.nan)
            else:
                # val stats
                rmse_val = mean_squared_error(y_val.reshape(-1, 1), y_val_preds, squared=False,
                                              multioutput='raw_values')
                r2_val = r2_score(y_val.reshape(-1, 1), y_val_preds)
                rmses_val.append(rmse_val)
                r2s_val.append(r2_val)
                # train stats rmses_train, r2s_train,
                rmse_train = mean_squared_error(y_train.reshape(-1, 1), y_train_preds, squared=False,
                                                multioutput='raw_values')
                r2_train = r2_score(y_train.reshape(-1, 1), y_train_preds)
                rmses_train.append(rmse_train)
                r2s_train.append(r2_train)

            # Update counter
            global_variables.inner_cv_loop += 1

        # ---- Inner CV loop finished
        global_variables.outer_test_loop += 1
        #print(
        #    f'Outer loop {global_variables.outer_test_loop} - with {global_variables.inner_cv_loop} inner loop, testing n epochs: {n_epochs_}, best at: {df_bestEpoch}')
        df_bestEpoch = None
        # Check if the trial should be pruned
        # ---- Optuna pruning
        if dict_train_params['optuna_metric'] == 'rmse':
            varOptuna = np.mean(rmses_val)
        elif dict_train_params['optuna_metric'] == 'r2':
            varOptuna = np.mean(r2s_val)

        trial.report(varOptuna, global_variables.outer_test_loop)  # report mse
        if trial.should_prune():  # let optuna decide whether to prune
            # save configuration and performances in a file
            df_report = pd.DataFrame([[trial.number, '@outer_loop' + str(global_variables.outer_test_loop),
                                       hp_dic['lr'], np.mean(rmses_train), np.mean(r2s_train),
                                       np.mean(rmses_val), np.mean(r2s_val), np.mean(rmses_test), np.mean(r2s_test),
                                       np.NAN,
                                       nbunits_conv_, kernel_size_, pool_size_, strides_, pyramid_bins_, dropout_rate_, nb_fc_,
                                       nunits_fc_, n_epochs_, batch_size_]],
                                     columns=['Trial', 'Pruned', 'lr', 'av_rmse_train', 'av_r2_train',
                                              'av_rmse_val', 'av_r2_val', 'av_rmse_test', 'av_r2_test',
                                              'av_r2_within_test',
                                              'nbunits_conv', 'kernel_size', 'pool_size', 'strides', 'pyramid_bins', 'dropout_rate',
                                              'n_fc', 'nunits_fc', 'n_epochs', 'batch_size'])
            if os.path.exists(fn_report):
                df_report.to_csv(fn_report, mode='a', header=False)
            else:
                df_report.to_csv(fn_report)
            raise optuna.exceptions.TrialPruned()

        # From the above I have validation statistics
        # ---- Now fit the model on training and validation data
        subset_bool = [x in train_val_i for x in groups]
        Xt_train, Xv_train, y_train = Xt_[subset_bool, :, :, :], region_ohe[subset_bool, :], y[subset_bool]
        if data_augmentation:
            Xt_train, Xv_train, y_train = generator.generate(Xt_train.shape[1], subset_bool)
        # ---- Normalizing the data per band
        min_per_t, max_per_t = readingsits2D.computingMinMax(Xt_train, per=0)
        # Normalise training set
        Xt_train = readingsits2D.normalizingData(Xt_train, min_per_t, max_per_t)
        Xt_test = readingsits2D.normalizingData(Xt_test, min_per_t, max_per_t)
        # Normalise ys
        transformer_y = MinMaxScaler().fit(y_train.reshape(-1, 1))
        ys_train = transformer_y.transform(y_train.reshape(-1, 1))
        # Compile and fit
        if model_type == '2DCNN_SISO':
            model, y_val_preds, bestEpoch = cv_Model(model, {'ts_input': Xt_train}, ys_train,
                                                     {'ts_input': None}, None,
                                                     out_model_file, n_epochs=n_epochs_,
                                                     batch_size=batch_size_,
                                                     learning_rate=learning_rate_,
                                                     beta_1=dict_train_params['beta_1'],
                                                     beta_2=dict_train_params['beta_2'],
                                                     decay=dict_train_params['decay'])
            X_test = {'ts_input': Xt_test}
        elif model_type == '2DCNN_MISO':
            model, y_val_preds, bestEpoch = cv_Model(model, {'ts_input': Xt_train, 'v_input': Xv_train}, ys_train,
                                                     {'ts_input': None, 'v_input': None}, None,
                                                     out_model_file, n_epochs=n_epochs_,
                                                     batch_size=batch_size_,
                                                     learning_rate=learning_rate_,
                                                     beta_1=dict_train_params['beta_1'],
                                                     beta_2=dict_train_params['beta_2'],
                                                     decay=dict_train_params['decay'])
            X_test = {'ts_input': Xt_test, 'v_input': Xv_test}
        # Now make prediction using all data in training and number of epochs from df_bestEpoch
        y_test_preds = model.predict(x=X_test)
        y_test_preds = transformer_y.inverse_transform(y_test_preds)

        out_test = np.concatenate([y_test.reshape(-1, 1), y_test_preds], axis=1)
        out_details = np.expand_dims(region_id[groups == test_i].T, axis=1)
        if not isinstance(df_details, np. ndarray): #df_details == None:
            df_details = np.concatenate([out_details, (np.ones_like(out_details) * test_i)], axis=1)
            df_test = out_test
        else:
            df_details = np.concatenate(
                [df_details, np.concatenate([out_details, (np.ones_like(out_details) * test_i)], axis=1)], axis=0)
            df_test = np.concatenate([df_test, out_test], axis=0)
        rmse_test = mean_squared_error(y_test.reshape(-1, 1), y_test_preds, squared=False, multioutput='raw_values')
        r2_test = r2_score(y_test.reshape(-1, 1), y_test_preds)
        rmses_test.append(rmse_test)
        r2s_test.append(r2_test)

    # test loop ended
    # Compute by cv folder average statistics (all excluding r2 test wich is compute in plotting)
    av_rmse_val = np.mean(rmses_val)
    av_r2_val = np.mean(r2s_val)
    av_rmse_test = np.mean(rmses_test)

    out_plot.plot_val_test_predictions_with_details(df_val, df_test, av_rmse_val, r2s_val, av_rmse_test, r2s_test,
                                                    xlabels, ylabels, df_details,
                                                    filename_val=fn_fig_val, filename_test=fn_fig_test)
    # Save CV results
    df_out = np.concatenate([df_details, df_test], axis=1)
    df_pd_out = pd.DataFrame(df_out, columns=['ASAP1_ID', 'Year', 'Observed', 'Predicted'])
    df_pd_out.to_csv(fn_cv_test, index=False)

    # Compute R2 within (avg of by AU temporal R2
    # Compute the mean of the tempral R2 computed by AU
    def r2_au(g):
        x = g['Observed']
        y = g['Predicted']
        # return metrics.r2_score(g['yLoo_true'], g['yLoo_pred'])
        return r2_score(x, y)

    r2within_test = df_pd_out.groupby('ASAP1_ID').apply(r2_au).mean()
    # save configuration and performances in a file
    df_report = pd.DataFrame([[trial.number, 'no', hp_dic['lr'], np.mean(rmses_train), np.mean(r2s_train),
                               av_rmse_val, av_r2_val, av_rmse_test, np.mean(r2s_test), r2within_test,
                               nbunits_conv_, kernel_size_, pool_size_, strides_, pyramid_bins_, dropout_rate_, nb_fc_, nunits_fc_,
                               n_epochs_, batch_size_]],
                             columns=['Trial', 'Pruned', 'lr', 'av_rmse_train', 'av_r2_train', 'av_rmse_val',
                                      'av_r2_val', 'av_rmse_test', 'av_r2_test', 'av_r2_within_test',
                                      'nbunits_conv', 'kernel_size', 'pool_size', 'strides', 'pyramid_bins', 'dropout_rate', 'n_fc',
                                      'nunits_fc', 'n_epochs', 'batch_size'])

    if os.path.exists(fn_report):
        df_report.to_csv(fn_report, mode='a', header=False)
    else:
        df_report.to_csv(fn_report)

    if dict_train_params['optuna_metric'] == 'rmse':
        return av_rmse_val
    elif dict_train_params['optuna_metric'] == 'r2':
        return av_r2_val



def run_wandb(args, month, input_size, trial, da_label, fn_asapID2AU, fn_stats90):
    # 1. Start a W&B run
    wandb.init(project=cst.wandb_project, entity=cst.wandb_entity, reinit=True,
               group=f'{args.target}C{crop_n}M{month}SZ{input_size}', config=trial.params,
               name=f'{args.target}-{model_type}-C{crop_n}-M{month}-{args.normalisation}-{da_label}',
               notes=f'Performance of a 2D CNN model for {args.target} forecasting in Algeria for'
                     f'crop ID {crop_n}.')

    # 2. Save model inputs and hyperparameters
    wandb.config.update({'model_type': model_type,
                         'crop_n': crop_n,
                         'month': month,
                         'norm': args.normalisation,
                         'target': args.target,
                         'n_epochs': dict_train_params['N_EPOCHS'],
                         'batch_size': dict_train_params['BATCH_SIZE'],
                         'n_trials': dict_train_params['N_TRIALS'],
                         'input_size': input_size
                         })

    # Evaluate best model on test set
    fn_csv_best = [x for x in (global_variables.dir_tgt / 'best_model').glob('*.csv')][0]
    res_i = mod_eval.model_evaluation(fn_csv_best, crop_n, month, model_type, fn_asapID2AU, fn_stats90)
    # 3. Log metrics over time to visualize performance
    wandb.log({"crop_n": crop_n,
               "month": month,
               "R2_p": res_i.R2_p.to_numpy()[0],
               "MAE_p": res_i.MAE_p.to_numpy()[0],
               "rMAE_p": res_i.rMAE_p.to_numpy()[0],
               "ME_p": res_i.ME_p.to_numpy()[0],
               "RMSE_p": res_i.RMSE_p.to_numpy()[0],
               "rRMSE_p": res_i.rRMSE_p.to_numpy()[0],
               "Country_R2_p": res_i.Country_R2_p.to_numpy()[0],
               "Country_MAE_p": res_i.Country_MAE_p.to_numpy()[0],
               "Country_ME_p": res_i.Country_ME_p.to_numpy()[0],
               "Country_RMSE_p": res_i.Country_RMSE_p.to_numpy()[0],
               "Country_rRMSE_p": res_i.Country_rRMSE_p.to_numpy()[0],
               "Country_FQ_rRMSE_p": res_i.Country_FQ_rRMSE_p.to_numpy()[0],
               "Country_FQ_RMSE_p": res_i.Country_FQ_RMSE_p.to_numpy()[0]
               })

    wandb.finish()


# -----------------------------------------------------------------------
if __name__ == "__main__":
    main()