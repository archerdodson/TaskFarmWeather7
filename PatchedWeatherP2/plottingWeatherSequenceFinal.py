### Metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.transforms import Bbox
import cartopy.crs as ccrs
import xarray as xr
# import and set up the typeguard
from typeguard.importhook import install_import_hook

from src.nn import ConditionalGenerativeModel, createGenerativeFCNN, InputTargetDataset, \
    UNet2D, DiscardWindowSizeDim, get_predictions_and_target, createGenerativeGRUNN, DiscardNumberGenerationsInOutput, \
    createGRUNN, createFCNN
from src.scoring_rules import EnergyScore, KernelScore, SignatureKernel, EnergyScorePath, VariogramScore, PatchedScoringRule, estimate_score_chunks
from src.utils import load_net, estimate_bandwidth_timeseries, lorenz96_variogram, def_loader_kwargs, \
    weatherbench_variogram_haversine
from src.parsers import parser_predict, define_masks, nonlinearities_dict, setup
from src.calibration import calibration_error, R2, rmse, plot_metrics_params, calibration_error_weighted, rmse_error_weighted, r2_error_weighted, CRPS, CRPS_weighted
from src.weatherbench_utils import load_weatherbench_data
from src.weatherbench_utils import load_weatherbench_data, convert_tensor_to_da, plot_map_ax

###############################
# HARDCODED CONFIGURATION
###############################
# Example values, adapt to your needs:
model = 'WeatherBench'
method = 'SR'
scoring_rule = 'SignatureKernel'
kernel = 'RBFtest'  ##??
patched = False
base_measure = 'normal'
root_folder = 'results'         # Where results are stored
model_folder = 'nets'           # Subfolder for models
datasets_folder = 'results/lorenz96/datasets/'
nets_folder = 'results/nets/'
weatherbench_data_folder = "../geopotential_500_5.625deg"
weatherbench_small = False

#name_postfix = '_mytrainedmodelEnergyScore' ##Change this
name_postfix = '_mytrainedmodelSignatureKernel_lr3e-04' ##Change this
training_ensemble_size = 3  #3/10
prediction_ensemble_size = 3 ##3/10
prediction_length = 2  

weights = np.array([0.07704437, 0.23039114, 0.38151911, 0.52897285, 0.67133229,
       0.80722643, 0.93534654, 1.05445875, 1.16341595, 1.26116882,
       1.34677594, 1.41941287, 1.47838008, 1.52310968, 1.55317091,
       1.56827425, 1.56827425, 1.55317091, 1.52310968, 1.47838008,
       1.41941287, 1.34677594, 1.26116882, 1.16341595, 1.05445875,
       0.93534654, 0.80722643, 0.67133229, 0.52897285, 0.38151911,
       0.23039114, 0.07704437])
ensemble_size = prediction_ensemble_size
unet_noise_method = 'sum'  # or 'concat', etc., if relevant
unet_large = True

lr = 0.1
lr_c = 0.0
batch_size = 1
no_early_stop = True
critic_steps_every_generator_step = 1

save_plots = True
cuda = True
load_all_data_GPU = True

nonlinearity = 'leaky_relu'
data_size = torch.Size([10, 32, 64])              # For Lorenz63, typically data_size=1 or 3
auxiliary_var_size = 1
seed = 0

plot_start_timestep = 0
plot_end_timestep = 100

gamma = None
gamma_patched = None
patch_size = 16
no_RNN = False
hidden_size_rnn = 32

save_pdf = True

save_pdf = True

compute_patched = model in ["lorenz96", ]

model_is_weatherbench = model == "WeatherBench"

nn_model = "unet" if model_is_weatherbench else ("fcnn" if no_RNN else "rnn")
print(nn_model)

method_is_gan = False



# datasets_folder, nets_folder, data_size, auxiliary_var_size, name_postfix, unet_depths, patch_size, method_is_gan, hidden_size_rnn = \
#     setup(model, root_folder, model_folder, datasets_folder, data_size, method, scoring_rule, kernel, patched,
#           patch_size, training_ensemble_size, auxiliary_var_size, critic_steps_every_generator_step, base_measure, lr,
#           lr_c, batch_size, no_early_stop, unet_noise_method, unet_large, nn_model, hidden_size_rnn)

model_name_for_plot = {"lorenz": "Lorenz63",
                       "lorenz96": "Lorenz96",
                       "WeatherBench": "WeatherBench"}[model]

string = f"Test {method} network for {model} model"
if not method_is_gan:
    string += f" using {scoring_rule} scoring rule"
print(string)

# # --- data handling ---
# if not model_is_weatherbench:
#     input_data_test = torch.load(datasets_folder + "test_x.pty",  weights_only=True)
#     target_data_test = torch.load(datasets_folder + "test_y.pty", weights_only=True)
#     input_data_val = torch.load(datasets_folder + "val_x.pty", weights_only=True)
#     target_data_val = torch.load(datasets_folder + "val_y.pty", weights_only=True)
#     scaling_data = target_data_test[:,0,0]
#     scaling_mean = scaling_data.mean()
#     scaling_std = scaling_data.std()

#     window_size = input_data_test.shape[1]

#     # create the test loaders; these are unused for the moment.
#     # dataset_val = InputTargetDataset(input_data_val, target_data_val, "cpu" )#if cuda and load_all_data_GPU else "cpu")
#     # dataset_test = InputTargetDataset(input_data_test, target_data_test,
#     #                                   "cpu" )#if cuda and load_all_data_GPU else "cpu")
# else:
print("Load weatherbench dataset...")
dataset_train, dataset_val, dataset_test = load_weatherbench_data(weatherbench_data_folder, cuda, load_all_data_GPU,
                                                            return_test=True,
                                                            weatherbench_small=weatherbench_small, predictionlength=prediction_length)
print("Loaded")
print("Validation set size:", len(dataset_val))
print("Test set size:", len(dataset_test))
print("find mean and std here")

loader_kwargs = def_loader_kwargs(cuda, load_all_data_GPU)
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **loader_kwargs)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, **loader_kwargs)


# loader_kwargs.update(loader_kwargs_2)  # if you want to add other loader arguments

#data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **loader_kwargs)
#data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, **loader_kwargs)

wrap_net = True
# create generative net:
unet_depths = (32, 64, 128, 256)
if nn_model == "unet":
        # select the noise method here:
        inner_net = UNet2D(in_channels=data_size[0], out_channels=1, noise_method=unet_noise_method,
                           number_generations_per_forward_call=prediction_ensemble_size, conv_depths=unet_depths)
        if unet_noise_method in ["sum", "concat"]:
            # here we overwrite the auxiliary_var_size above, as there is a precise constraint
            downsampling_factor, n_channels = inner_net.calculate_downsampling_factor()
            if weatherbench_small:
                auxiliary_var_size = torch.Size(
                    [n_channels, 16 // downsampling_factor, 16 // downsampling_factor])
            else:
                auxiliary_var_size = torch.Size(
                    [n_channels, data_size[1] // downsampling_factor, data_size[2] // downsampling_factor])
        elif unet_noise_method == "dropout":
            wrap_net = False  # do not wrap in the conditional generative model
if wrap_net:
    net = load_net(nets_folder + f"net{name_postfix}.pth", ConditionalGenerativeModel, inner_net,
                    size_auxiliary_variable=auxiliary_var_size, base_measure=base_measure,
                    number_generations_per_forward_call=prediction_ensemble_size, seed=seed + 1)
else:
    net = load_net(nets_folder + f"net{name_postfix}.pth", DiscardWindowSizeDim, inner_net)

if cuda:
    net.cuda()


# def make_prediction(net, inputs, prediction_length):
#     """
#     a function to create l (= prediction_length) step ahead predictions given the input data

#     """

#     device = torch.device('cuda' if cuda else 'cpu')
#     inputs = inputs.to(device)

#     inputs = torch.unsqueeze(inputs, 0)
#     #print(inputs.shape)
#     initialwindow = inputs  #  (1,10,1)
#     outputs1 = net(inputs)   # 1, 7, 1
#     #print(outputs1.shape)
#     ensemble_length = outputs1.shape[1]

#     # Cloning making list for grad 7 list of 1,10,1
#     windowensemble = [initialwindow.clone() for _ in range(ensemble_length)]

#     forecasts = []

#     for step in range(prediction_length):
#         onesteps = []

#         for e in range(ensemble_length):
#             eoutput = net(windowensemble[e])[:,e,:] #pick model e out

#             #move ensemble window down, replace with model output
#             #pop off output
#             eoutput = eoutput.unsqueeze(1)  #add dimension to line up without broadcasting
#             shifted = torch.cat([windowensemble[e][:, 1:, :], eoutput], dim=1)  # 1,9,1 + 1,1,1    1,10,1
#             windowensemble[e] = shifted

#             onesteps.append(eoutput) 

#         onesteps = torch.cat(onesteps, dim=0)

#         if step == 0:
#             forecasts = onesteps 
#         else:
#             forecasts = torch.cat((forecasts, onesteps), dim=1) 
#     return forecasts

# def makepredictionsequence(net, inputs, prediction_length):
    
#     forecasts = []
#     for entry in range(inputs.shape[0]):
#         #print('yoloop')
#         forecastentry = make_prediction(net,inputs[entry,:,:],prediction_length)
#         #print(forecastentry.shape)
#         forecasts.append(forecastentry)

#     forecasts = torch.stack(forecasts, dim=0)
#     print(forecasts.shape)
#     return(forecasts)



# output_size = data_size
# gru_layers = 1
# gru_hidden_size = hidden_size_rnn
# inner_net = createGenerativeGRUNN(data_size=data_size, gru_hidden_size=gru_hidden_size,
#                                 noise_size=auxiliary_var_size,
#                                 output_size=output_size, hidden_sizes=None, gru_layers=gru_layers,
#                                 nonlinearity=nonlinearities_dict[nonlinearity])()
# if wrap_net:
#     net = load_net(nets_folder + f"net{name_postfix}.pth", ConditionalGenerativeModel, inner_net,
#                 size_auxiliary_variable=auxiliary_var_size, base_measure=base_measure,
#                 number_generations_per_forward_call=prediction_ensemble_size, seed=seed + 1)
if cuda:
    net.cuda()

# --- predictions ---
# predict all the different elements of the test set and create plots.
# can directly feed through the whole test set for now; if it does not work well then, I will batch it.
# print(input_data_val.shape) #36,10,1     10 windows
# print(input_data_test.shape)
# print(target_data_val.shape) #36,9,1  True
# print(target_data_test.shape)


##Testing ##
with torch.no_grad():
    forecastslargetest = []
    predictionslargetest = []
    for batch_idx, (data, target) in enumerate(data_loader_test):

        if batch_idx>10:
            break
        batchsize = data.shape[0]
        datasizes = data.shape
        #print(datasizes)
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
        
        outputs1 = net(*data)
        ensemble_length = outputs1.shape[1]

        batch_loss = torch.zeros(1,device = "cuda" if cuda else "cpu")

        for batch in range(batchsize):
            databatch = torch.unsqueeze(data[0][batch,:,:],0)
            targetbatch = torch.unsqueeze(target[batch,:,:],0)

            initialwindow = databatch  #  (1,10,1)

            # Cloning making list for grad 7 list of 1,10,1
            windowensemble = [initialwindow.clone() for _ in range(ensemble_length)]

            forecasts = []

            #net.resetseed()

            for step in range(prediction_length):
                onesteps = []

                for e in range(ensemble_length):
                    eoutput = net(windowensemble[e])[:,e,:] #pick model e out

                    #move ensemble window down, replace with model output
                    #pop off output
                    eoutput = eoutput.unsqueeze(1)  #add dimension to line up without broadcasting
                    shifted = torch.cat([windowensemble[e][:, 1:, :], eoutput], dim=1)  # 1,9,1 + 1,1,1    1,10,1
                    windowensemble[e] = shifted

                    onesteps.append(eoutput) 

                onesteps = torch.cat(onesteps, dim=0)

                if step == 0:
                    forecasts = onesteps 
                else:
                    forecasts = torch.cat((forecasts, onesteps), dim=1)

            forecastslargetest.append(forecasts)
            predictionslargetest.append(target)
    forecastslargetest = torch.stack(forecastslargetest, dim=0)
    predictionslargetest = torch.stack(predictionslargetest, dim=0)
    print('test')
    print(forecastslargetest.shape)
    print(predictionslargetest.shape)

forecastslargetest = forecastslargetest
predictionslargetest = predictionslargetest.squeeze(1)
print(forecastslargetest.shape) #(11?, ensemble=5,path length=10, lat=32, long=64, 1)
print(predictionslargetest.shape) #(11?,2, lat=32, long=64, 1)
print('yo')
####Pick a prediction and part of pathlength

#date = "2018-01-01"
prediction = forecastslargetest[10,1,:,:,:,:].unsqueeze(0)
print(prediction.shape)
realization = predictionslargetest[10,:,:,:,:].unsqueeze(0)
print(realization.shape)
#Each is torch.Size([1, 10, 32, 64, 1])
#zero corresponds to 2018-01-11
date = "2018-01-11"
# predict for a given date and create the plot
with torch.no_grad():
    # obtain the target and context for the specified timestring
    timestring = date + "T12:00:00.000000000"
    _, realization1 = dataset_test.select_time(timestring)

# print('yo')
# print(realization)
# print('yo1')
# print(realization1)


prediction = prediction.cpu()
realization = realization.cpu()
realizationtest = realization1
# ... [Your existing code until the plotting section] ...


# Convert each time step to DataArray
da_realizations = []
for i in range(realization.shape[1]):  # Loop over time steps
    da_realizations.append(convert_tensor_to_da(realization[0, i], realizationtest))

da_predictions = []
for i in range(prediction.shape[1]):   # Loop over time steps
    da_predictions.append(convert_tensor_to_da(prediction[0, i], realizationtest))

if save_plots:
    global_projection = False
    n_time_steps = prediction_length  # Path length

    # --- Plot 1: Absolute Values (2 rows: true & predicted paths) ---
    kwargs_abs = dict(ncols=n_time_steps, nrows=2, figsize=(25, 5),
                      subplot_kw=dict(projection=ccrs.PlateCarree(), facecolor="gray"))
    
    fig, axes = plt.subplots(**kwargs_abs)
    
    # Global vmin/vmax for color consistency
    all_data = np.concatenate([r.values for r in da_realizations] + [p.values for p in da_predictions])
    vmax, vmin = np.max(all_data), np.min(all_data)
    
    # Row 0: True realizations
    for i in range(n_time_steps):
        plot_map_ax(da_realizations[i][:, :, 0], 
                    title=f"True t={i}", 
                    ax=axes[0, i], 
                    global_projection=global_projection, 
                    vmax=vmax, vmin=vmin)
    
    # Row 1: Predictions
    for i in range(n_time_steps):
        p = plot_map_ax(da_predictions[i][:, :, 0], 
                        title=f"Pred t={i}", 
                        ax=axes[1, i], 
                        global_projection=global_projection, 
                        vmax=vmax, vmin=vmin)
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(p, cax=cbar_ax)
    fig.suptitle(f"Z500, {date}", size=20, y=1.02)
    plt.savefig(nets_folder + f"map_absolute{name_postfix}." + ("pdf" if save_pdf else "png"), bbox_inches="tight")

    # --- Plot 2: Differences (Prediction - True) ---
    kwargs_diff = dict(ncols=n_time_steps, nrows=1, figsize=(25, 2.5),
                       subplot_kw=dict(projection=ccrs.PlateCarree(), facecolor="gray"))
    
    fig, axes = plt.subplots(**kwargs_diff)
    differences = [da_predictions[i] - da_realizations[i] for i in range(n_time_steps)]
    
    # Global vmin/vmax for differences
    diff_max = max(diff.max() for diff in differences)
    diff_min = min(diff.min() for diff in differences)
    
    for i in range(n_time_steps):
        p = plot_map_ax(differences[i][:, :, 0], 
                        title=f"Diff t={i}", 
                        ax=axes[i], 
                        global_projection=global_projection, 
                        vmax=diff_max, vmin=diff_min)
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(p, cax=cbar_ax)
    fig.suptitle(f"Z500, Prediction - Realization, {date}", size=20, y=1.02)
    plt.savefig(nets_folder + f"map_differences{name_postfix}." + ("pdf" if save_pdf else "png"), bbox_inches="tight")

    # --- Plot 3: Anomalies (Centered on Temporal Mean) ---
    # Compute temporal means
    temporal_mean_true = sum(da_realizations) / n_time_steps  # Mean over time
    temporal_mean_pred = sum(da_predictions) / n_time_steps
    
    # Compute anomalies
    anomalies_true = [da_realizations[i] - temporal_mean_true for i in range(n_time_steps)]
    anomalies_pred = [da_predictions[i] - temporal_mean_pred for i in range(n_time_steps)]
    
    fig, axes = plt.subplots(**kwargs_abs)  # Reuse 2x10 layout
    
    # Global vmin/vmax for anomalies
    all_anom = np.concatenate([a.values for a in anomalies_true + anomalies_pred])
    anom_max, anom_min = np.max(all_anom), np.min(all_anom)
    
    # Row 0: True anomalies
    for i in range(n_time_steps):
        plot_map_ax(anomalies_true[i][:, :, 0], 
                    title=f"True Anom t={i}", 
                    ax=axes[0, i], 
                    global_projection=global_projection, 
                    vmax=anom_max, vmin=anom_min)
    
    # Row 1: Prediction anomalies
    for i in range(n_time_steps):
        p = plot_map_ax(anomalies_pred[i][:, :, 0], 
                        title=f"Pred Anom t={i}", 
                        ax=axes[1, i], 
                        global_projection=global_projection, 
                        vmax=anom_max, vmin=anom_min)
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(p, cax=cbar_ax)
    fig.suptitle(f"Z500, Anomalies from Temporal Mean, {date}", size=20, y=1.02)
    plt.savefig(nets_folder + f"map_anomalies{name_postfix}." + ("pdf" if save_pdf else "png"), bbox_inches="tight")

    # --- Plot 4: Temporal Mean & Std of Predictions ---
    kwargs_mean_std = dict(ncols=2, nrows=1, figsize=(12, 4),
                           subplot_kw=dict(projection=ccrs.PlateCarree(), facecolor="gray"))
    
    fig, axes = plt.subplots(**kwargs_mean_std)
    
    # Temporal std (over time steps)
    pred_stack = xr.concat(da_predictions, dim="time")
    temporal_std_pred = pred_stack.std(dim="time")
    
    plot_map_ax(temporal_mean_pred[:, :, 0], 
                title="Temporal Mean", 
                ax=axes[0], 
                global_projection=global_projection)
    
    p = plot_map_ax(temporal_std_pred[:, :, 0], 
                    title="Temporal Std", 
                    ax=axes[1], 
                    global_projection=global_projection)
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(p, cax=cbar_ax)
    fig.suptitle(f"Z500, Prediction Temporal Stats, {date}", size=20, y=1.02)
    plt.savefig(nets_folder + f"map_temporal_stats{name_postfix}." + ("pdf" if save_pdf else "png"), bbox_inches="tight")