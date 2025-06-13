### Metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.transforms import Bbox
# import and set up the typeguard
from typeguard.importhook import install_import_hook

from src.nn import ConditionalGenerativeModel, createGenerativeFCNN, InputTargetDataset, \
    UNet2D, DiscardWindowSizeDim, get_predictions_and_target, createGenerativeGRUNN, DiscardNumberGenerationsInOutput, \
    createGRUNN, createFCNN
from src.scoring_rules import EnergyScore, KernelScore, SignatureKernel, EnergyScorePath, VariogramScore, PatchedScoringRule, estimate_score_chunks
from src.utils import load_net, estimate_bandwidth_timeseries, lorenz96_variogram, def_loader_kwargs, \
    weatherbench_variogram_haversine
from src.parsers import parser_predict, define_masks, nonlinearities_dict, setup
from src.calibration import calibration_error, R2, rmse, plot_metrics_params, calibration_error_weighted, rmse_error_weighted, r2_error_weighted, CRPS, CRPS_weighted, RQE_weighted, relative_quantile_error
from src.weatherbench_utils import load_weatherbench_data

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
name_postfix = '_mytrainedmodelSignatureKernel' ##Change this
training_ensemble_size = 3  #3/10
prediction_ensemble_size = 10 ##3/10
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
                                                            weatherbench_small=weatherbench_small)
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
        batchsize = data.shape[0]
        datasizes = data.shape
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

            net.resetseed()

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

forecastslargetest = forecastslargetest.squeeze(-1)
predictionslargetest = predictionslargetest.squeeze(-1).squeeze(1)
print(forecastslargetest.shape)
print(predictionslargetest.shape)
        # print(batch_idx)
        # print(datasizes)
        # print(target.shape)
        # print(forecasts.shape)



# for batch_idx, (data, target) in enumerate(data_loader_test):
#     print(batch_idx)
#     print(data.shape)
#     print(target.shape)

#print(breakd)
# with torch.no_grad():
#         print('test')
#         #Clearly needs to be fixed
#         #print("yo1")
# # shape (n_val, ensemble_size, lon, lat, n_fields)
#         # predictions_val, target_data_val = get_predictions_and_target(data_loader_val, net, cuda)
#         # predictions_test, target_data_test = get_predictions_and_target(data_loader_test, net, cuda)

#         predictions_val = makepredictionsequence(net, data_loader_val,prediction_length)
#         predictions_test = makepredictionsequence(net, data_loader_test,prediction_length)
#         # predictions_val = net(input_data_val)  # shape (n_val, ensemble_size, data_size)
#         # predictions_test = net(input_data_test)  # shape (n_test, ensemble_size, data_size)
# print(predictions_val.shape) #Current output is 36,20,1 #We need 36,20,9,1
# print(predictions_test.shape)
print('yo')

#36,20,9,1
#38,20,9,1



# if method != "regression":
#     # --- scoring rules ---

#     # if gamma is None:
#     #     print("Compute gamma...")
#     #     gamma = estimate_bandwidth_timeseries(target_data_val, return_values=["median"])
#     #     print(f"Estimated gamma: {gamma:.4f}")
#     # if gamma_patched is None and compute_patched:
#     #     # determine the gamma using the first patch only. This assumes that the values of the variables
#     #     # are roughly the same in the different patches.
#     #     gamma_patched = estimate_bandwidth_timeseries(target_data_val[:, masks[0]], return_values=["median"])
#     #     print(f"Estimated gamma patched: {gamma_patched:.4f}")

#     # instantiate SRs; each SR takes as input: (net_output, target)
#     sr = SignatureKernel() ##### Could replace with Signature Kernel ##MSEScore ##SignatureKernel #EnergyScorePath
#     #loss_fn = sr_instance.estimate_score_batch
#     #scoring_rule = "SignatureKernel" #SignatureKernel
#     # kernel_gaussian_sr = KernelScore(sigma=gamma)
#     # kernel_rat_quad_sr = KernelScore(kernel="rational_quadratic", alpha=gamma ** 2)
#     energy_sr = EnergyScorePath()

#     # variogram = None
#     # if model in ["lorenz96", ]:
#     #     variogram = lorenz96_variogram(data_size)
#     # elif model == "WeatherBench":
#     #     # variogram = weatherbench_variogram(weatherbench_small=weatherbench_small)
#     #     variogram = weatherbench_variogram_haversine(weatherbench_small=weatherbench_small)
#     # if variogram is not None and cuda:
#     #     variogram = variogram.cuda()

#     # variogram_sr = VariogramScore(variogram=variogram)

#     # if compute_patched:
#     #     # patched SRs:
#     #     kernel_gaussian_sr_patched = PatchedScoringRule(KernelScore(sigma=gamma_patched), masks)
#     #     kernel_rat_quad_sr_patched = PatchedScoringRule(
#     #         KernelScore(kernel="rational_quadratic", alpha=gamma_patched ** 2),
#     #         masks)
#     #     energy_sr_patched = PatchedScoringRule(energy_sr, masks)

#     # -- out of sample score --
#     scaling_mean= 0
#     scaling_std = 1

#     with torch.no_grad():
#         string = ""
#         for name, predictions, target in zip(["VALIDATION", "TEST"], [predictions_val, predictions_test],
#                                              [target_data_val, target_data_test]):
#             string += name + "\n"
#             print(name.shape)
#             print(predictions.shape) #36,20,9,1
#             print(target.shape) #36,9,1
#             print("yo2")

#             SigRBFscore = estimate_score_chunks(sr, predictions, target,scaling_mean,scaling_std, chunk_size=1)
#             #kernel_rat_quad_score = estimate_score_chunks(kernel_rat_quad_sr, predictions, target)
#             energy_score = estimate_score_chunks(energy_sr, predictions, target,scaling_mean,scaling_std, chunk_size=1)
#             #variogram_score = estimate_score_chunks(variogram_sr, predictions, target, chunk_size=8)

#             string += f"Whole data scores: \nEnergy score: {energy_score:.2f}, " \
#                       f"Gaussian Kernel score {SigRBFscore:.2f}\n" 
#                     #   f" Rational quadratic Kernel score {kernel_rat_quad_score:.2f}, " \
#                     #   f"Variogram score {variogram_score:.2f}\n"

#         print(string)

with torch.no_grad():
    # -- calibration metrics --
    # target_data_test shape (n_test, data_size)
    # predictions_test shape (n_test, ensemble_size, data_size)
    print("three metrics") 
    #print(target_data_test.shape) #38,9,1
    #print(predictions_test.shape) #38,20,9,1


    predictions_for_calibration = forecastslargetest.transpose(1, 0).transpose(3,2).cpu().detach().numpy()
    target_data_test_for_calibration = predictionslargetest.transpose(2,1).cpu().detach().numpy()
    print(predictions_for_calibration.shape)
    print(target_data_test_for_calibration.shape)

    predictions_for_calibrationarea = predictions_for_calibration.reshape(prediction_ensemble_size,354,32,64*prediction_length)
    target_data_test_for_calibrationarea = target_data_test_for_calibration.reshape(354, 32, 64*prediction_length)
    print('area')
    print(predictions_for_calibrationarea.shape)
    print(target_data_test_for_calibrationarea.shape)

    cal_err_values_weighted = calibration_error_weighted(predictions_for_calibrationarea, target_data_test_for_calibrationarea, weights)
    rmse_values_weighted = rmse_error_weighted(predictions_for_calibrationarea, target_data_test_for_calibrationarea, weights)
    r2_values_weighted = r2_error_weighted(predictions_for_calibrationarea, target_data_test_for_calibrationarea, weights)
    CRPS_values_weighted = CRPS_weighted(predictions_for_calibrationarea, target_data_test_for_calibrationarea, weights)
    RQE_values_weighted = RQE_weighted(predictions_for_calibrationarea, target_data_test_for_calibrationarea, weights)


    print('bad')
    predictions_for_calibration = predictions_for_calibration.reshape(prediction_ensemble_size,354,32*64*prediction_length)
    target_data_test_for_calibration = target_data_test_for_calibration.reshape(354, 32*64*prediction_length)
    print(predictions_for_calibration.shape)
    print(target_data_test_for_calibration.shape)
    data_size = target_data_test_for_calibration.shape[-1]

    #print(predictions_for_calibration.shape) #38,9,1
    #print(target_data_test_for_calibration.shape) #20,38,9,1
    cal_err_values = calibration_error(predictions_for_calibration, target_data_test_for_calibration)
    rmse_values = rmse(predictions_for_calibration, target_data_test_for_calibration)
    r2_values = R2(predictions_for_calibration, target_data_test_for_calibration)
    CRPS_values = CRPS(predictions_for_calibration, target_data_test_for_calibration)
    RQE_values = relative_quantile_error(predictions_for_calibrationarea, target_data_test_for_calibrationarea)

    string2 = f"Calibration metrics:\n"
    for i in range(data_size):
        string2 += f"x{i}: Cal. error {cal_err_values[i]:.4f}, RMSE {rmse_values[i]:.4f}, R2 {r2_values[i]:.4f}\n"
    string2 += f"\nAverage values: Cal. error {cal_err_values.mean():.4f}, RMSE {rmse_values.mean():.4f}, R2 {r2_values.mean():.4f}, NCRPS {CRPS_values.mean():.4f}, RQE {RQE_values.mean():.4f}\n"
    string2 += f"\nStandard deviation: Cal. error {cal_err_values.std():.4f}, RMSE {rmse_values.std():.4f}, R2 {r2_values.std():.4f}, NCRPS {CRPS_values.std():.4f}, RQE {RQE_values.std():.4f}\n\n"

    string2 += f"\nAverage values area: Cal. error {cal_err_values_weighted.mean():.4f}, RMSE {rmse_values_weighted.mean():.4f}, R2 {r2_values_weighted.mean():.4f}, NCRPS {CRPS_values_weighted.mean():.4f}, RQE {RQE_values_weighted.mean():.4f}\n"
    string2 += f"\nStandard deviation: Cal. error {cal_err_values_weighted.std():.4f}, RMSE {rmse_values_weighted.std():.4f}, R2 {r2_values_weighted.std():.4f}, NCRPS {CRPS_values_weighted.std():.4f}, RQE {RQE_values_weighted.std():.4f}\n\n"
    

    string2 += f"\nAverage values: Cal. error, RMSE, R2 \n"
    string2 += f"\n\t\t {cal_err_values.mean():.4f} $ \pm$ {cal_err_values.std():.4f} & {rmse_values.mean():.4f}  $ \pm$ {rmse_values.std():.4f} &  {r2_values.mean():.4f} $ \pm$ {r2_values.std():.4f} \\\\ \n"
    string2 += f"\n\t\t {cal_err_values.mean():.4f}  & {rmse_values.mean():.4f}  &  {r2_values.mean():.4f}  \\\\ \n"
    print(string2)

    # -- plots --
with torch.no_grad():
    # if model_is_weatherbench:
    #     # we visualize only the first 8 variables.
    #     variable_list = np.linspace(0, target_data_test.shape[-1] - 1, 8, dtype=int)
    #     predictions_test = predictions_test[:, :, variable_list]
    #     target_data_test = target_data_test[:, variable_list]
    #     predictions_for_calibration = predictions_for_calibration[:, :, variable_list]
    #     target_data_test_for_calibration = target_data_test_for_calibration[:, variable_list]
    predictions_test = predictions_test.reshape(99, prediction_ensemble_size,data_size)
    target_data_test = target_data_test.reshape(99, data_size)
    predictions_test_for_plot = predictions_test.cpu()
    target_data_test_for_plot = target_data_test.cpu()
    time_vec = torch.arange(len(predictions_test)).cpu()
    data_size = predictions_test_for_plot.shape[-1]

    # if model == "lorenz":
    #     var_names = [r"$y$"]
    # elif model == "WeatherBench":
    #     # todo write here the correct lon and lat coordinates!
    #     var_names = [r"$x_{}$".format(i + 1) for i in range(data_size)]
    # else:
    data_size = 3
    var_names = [r"$x_{}$".format(i + 1) for i in range(data_size)]

    # predictions: mean +- std
    label_size = 13
    if method != "regression":
        predictions_mean = torch.mean(predictions_test_for_plot, dim=1).detach().numpy()
        predictions_std = torch.std(predictions_test_for_plot, dim=1).detach().numpy()

        fig, ax = plt.subplots(nrows=data_size, ncols=1, sharex="col", figsize=(6.4, 3) if data_size == 1 else None)
        if data_size == 1:
            ax = [ax]
        for var in range(data_size):
            ax[var].plot(time_vec[plot_start_timestep:plot_end_timestep],
                         target_data_test_for_plot[plot_start_timestep:plot_end_timestep, var], ls="--",
                         color=f"C{var}")
            ax[var].plot(time_vec[plot_start_timestep:plot_end_timestep],
                         predictions_mean[plot_start_timestep:plot_end_timestep, var], ls="-", color=f"C{var}")
            ax[var].fill_between(
                time_vec[plot_start_timestep:plot_end_timestep], alpha=0.3, color=f"C{var}",
                y1=predictions_mean[plot_start_timestep:plot_end_timestep, var] -
                   predictions_std[plot_start_timestep:plot_end_timestep, var],
                y2=predictions_mean[plot_start_timestep:plot_end_timestep, var] +
                   predictions_std[plot_start_timestep:plot_end_timestep, var])
            ax[var].set_ylabel(var_names[var], size=label_size)

        ax[-1].set_xlabel("Integration time index")
        fig.suptitle(r"Mean $\pm$ std, " + model)
        # plt.show()
        if save_plots:
            plt.savefig(nets_folder + f"prediction{name_postfix}.png")
        plt.close()

    # predictions: median and 99% quantile region
    np_predictions = predictions_test_for_plot.detach().numpy()
    size = 99
    predictions_median = np.median(np_predictions, axis=1)
    if method != "regression":
        predictions_lower = np.percentile(np_predictions, 50 - size / 2, axis=1)
        predictions_upper = np.percentile(np_predictions, 50 + size / 2, axis=1)

    fig, ax = plt.subplots(nrows=data_size, ncols=1, sharex="col", figsize=(6.4, 3) if data_size == 1 else None)
    if data_size == 1:
        ax = [ax]
    for var in range(data_size):
        ax[var].plot(time_vec[plot_start_timestep:plot_end_timestep],
                     target_data_test_for_plot[plot_start_timestep:plot_end_timestep, var], ls="--", color=f"C{var}",
                     label="True")
        ax[var].plot(time_vec[plot_start_timestep:plot_end_timestep],
                     predictions_median[plot_start_timestep:plot_end_timestep, var], ls="-", color=f"C{var}",
                     label="Median forecast" if method != "regression" else "Forecast")
        if method != "regression":
            ax[var].fill_between(
                time_vec[plot_start_timestep:plot_end_timestep], alpha=0.3, color=f"C{var}",
                y1=predictions_lower[plot_start_timestep:plot_end_timestep, var],
                y2=predictions_upper[plot_start_timestep:plot_end_timestep, var], label="99% credible region")
        ax[var].set_ylabel(var_names[var], size=label_size)
        ax[var].tick_params(axis='both', which='major', labelsize=label_size)

    if data_size == 1:
        ax[0].legend(fontsize=label_size)

    ax[-1].set_xlabel(r"$t$", size=label_size)
    # fig.suptitle(f"Median and {size}% credible region, " + model_name_for_plot, size=title_size)
    # plt.show()

    if save_plots:
        # save the metrics in file
        text_file = open(nets_folder + f"test_losses{name_postfix}.txt", "w")
        text_file.write(string + "\n")
        text_file.write(string2 + "\n")
        text_file.close()
        # save the plot:

        if data_size == 1:
            bbox = Bbox(np.array([[0, -0.2], [6.1, 3]]))
        else:
            bbox = Bbox(np.array([[0, -0.2], [6.0, 4.8]]))
        plt.savefig(nets_folder + f"prediction_median{name_postfix}." + ("pdf" if save_pdf else "png"), dpi=400,
                    bbox_inches=bbox)
    plt.close()

    if not model_is_weatherbench:
        # metrics plots
        plot_metrics_params(cal_err_values, rmse_values, r2_values,
                            filename=nets_folder + f"metrics{name_postfix}.png" if save_plots else None)