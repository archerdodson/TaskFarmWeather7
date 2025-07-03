import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset
# import and set up the typeguard
# from typeguard.importhook import install_import_hook

# # comment these out when deploying:
# install_import_hook('src.nn')
# install_import_hook('src.scoring_rules')
# install_import_hook('src.utils')
# install_import_hook('src.weatherbench_utils')
# install_import_hook('src.unet_utils')

from src.nn import InputTargetDataset, UNet2D, fit, fit_adversarial, \
    ConditionalGenerativeModel, createGenerativeFCNN, createCriticFCNN, test_epoch, test_epochlongerpredictionbatch, PatchGANDiscriminator, \
    DiscardWindowSizeDim, get_target, LayerNormMine, createGenerativeGRUNN, createCriticGRUNN, \
    DiscardNumberGenerationsInOutput, createGRUNN, createFCNN
from src.scoring_rules import EnergyScore, SignatureKernelRBF, MSEScore, SignatureKernel, EnergyScorePath, KernelScore, VariogramScore, PatchedScoringRule, SumScoringRules, \
    ScoringRulesForWeatherBench, ScoringRulesForWeatherBenchPatched, LossForWeatherBenchPatched
from src.utils import plot_losses, save_net, save_dict_to_json, estimate_bandwidth_timeseries, lorenz96_variogram, \
    def_loader_kwargs, load_net, weight_for_summed_score, weatherbench_variogram_haversine, estimate_bandwidth_per_batch
from src.parsers import parser_train_net, define_masks, nonlinearities_dict, setup
from src.weatherbench_utils import load_weatherbench_data


#####################################################################
batch_size, ensemble_size, method, nn_model, data_size, auxiliary_var_size = 512, 3, 'SR', 'rnn', 1, 1 #mess around with model size
shuffletrue = True #Could be False?
hidden_size_rnn = 8
nonlinearities_dict, nonlinearity = {"relu": torch.nn.functional.relu, "tanhshrink": torch.nn.functional.tanhshrink,
                       "leaky_relu": torch.nn.functional.leaky_relu, "gelu": torch.nn.functional.gelu}, 'leaky_relu' #'leaky_relu standard #gelu? 

#Implement gelu

args_dict = {}
weight_decay, scheduler_gamma, lr, epochs, early_stopping, epochs_early_stopping_interval = 0, 1, 0.001, 1000, True, 20 #Original Learning rate 0.01 

model, scoring_rule = 'lorenz63', 'SignatureKernel' #Doesn't matter?
cuda, continue_training_net, start_epoch_early_stopping, use_tqdm, method_is_gan  = True, False, 200, True, False

base_measure, seed = 'normal', 0

datasets_folder = 'results/lorenz/datasets/'
nets_folder = "results/nets/"
model_is_weatherbench = False

# --- loss function ---
sr_instance = SignatureKernel() ##### Could replace with Signature Kernel ##MSEScore ##SignatureKernel
loss_fn = sr_instance.estimate_score_batch
scoring_rule = "SignatureKernel" #SignatureKernel
#name_postfix = "_mytrainedmodelEnergyScore"  # or something else descriptive
name_postfix = "_mytrainedmodelSignatureKernel"

# --- data handling ---
if not model_is_weatherbench:
    input_data_train = torch.load(datasets_folder + "train_x.pty", weights_only=True)#[:8]
    target_data_train = torch.load(datasets_folder + "train_y.pty",weights_only=True)#[:8]
    scaling_data = target_data_train[:,0,0]
    scaling_mean = scaling_data.mean()
    scaling_std = scaling_data.std()
    input_data_val = torch.load(datasets_folder + "val_x.pty",weights_only=True)
    target_data_val = torch.load(datasets_folder + "val_y.pty",weights_only=True)
    scaling_data_val= target_data_val[:,0,0]
    scaling_mean_val = scaling_data_val.mean()
    scaling_val_std = scaling_data_val.std()
    window_size = input_data_train.shape[1]
    prediction_length = target_data_train.shape[1]  
    #print(prediction_length)
    data_size = input_data_train.shape[-1]
    
    # create the train and val loaders:
    dataset_train = InputTargetDataset(input_data_train, target_data_train, "cuda")#,
                                      # "cuda" if cuda and load_all_data_GPU else "cpu")
    dataset_val = InputTargetDataset(input_data_val, target_data_val, "cuda") ###
                                     #, "cuda" if cuda and load_all_data_GPU else "cpu") 
else:
    dataset_train, dataset_val = load_weatherbench_data(weatherbench_data_folder, cuda, load_all_data_GPU,
                                                        weatherbench_small=weatherbench_small)
    len_dataset_train = len(dataset_train)
    len_dataset_val = len(dataset_val)
    print("Training set size:", len_dataset_train)
    print("Validation set size:", len_dataset_val)
    args_dict["len_dataset_train"] = len_dataset_train
    args_dict["len_dataset_val"] = len_dataset_val

# loader_kwargs = def_loader_kwargs(cuda, load_all_data_GPU)

# loader_kwargs.update(loader_kwargs_2)  # if you want to add other loader arguments


#gamma = estimate_bandwidth_per_batch(target_data_val, return_values=["median"])
#print(gamma) gamma = 9?

data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffletrue)
#data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)   #, **loader_kwargs)
if len(dataset_val) > 0:
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False) #, **loader_kwargs)
    if model_is_weatherbench:
        # obtain the target tensor to estimate the gamma for kernel SR:
        target_data_val = get_target(data_loader_val, cuda).flatten(1, -1)
else:
    data_loader_val = None

# # --- loss function ---
# sr_instance = SignatureKernel() ##### Could replace with Signature Kernel ##MSEScore ##SignatureKernel
# loss_fn = sr_instance.estimate_score_batch
# scoring_rule = "SignatureKernel" #SignatureKernel
# #name_postfix = "_mytrainedmodelEnergyScore"  # or something else descriptive
# #name_postfix = "_mytrainedmodelkernelrbf"


# --- defining the model using a generative net ---


wrap_net = True
number_generations_per_forward_call = ensemble_size if method == "SR" else 1
# create generative net:
if nn_model == "fcnn":
    input_size = window_size * data_size + auxiliary_var_size
    output_size = data_size
    hidden_sizes_list = [int(input_size * 1.5), int(input_size * 3), int(input_size * 3),
                            int(input_size * 0.75 + output_size * 3), int(output_size * 5)]
    inner_net = createGenerativeFCNN(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes_list,
                                        nonlinearity=nonlinearities_dict[nonlinearity])()
elif nn_model == "rnn":
    output_size = data_size
    gru_layers = 1
    gru_hidden_size = hidden_size_rnn
    inner_net = createGenerativeGRUNN(data_size=data_size, gru_hidden_size=gru_hidden_size,
                                        noise_size=auxiliary_var_size,
                                        output_size=output_size, hidden_sizes=None, gru_layers=gru_layers,
                                        nonlinearity=nonlinearities_dict[nonlinearity])()
elif nn_model == "unet":
    # select the noise method here:
    inner_net = UNet2D(in_channels=data_size[0], out_channels=1, noise_method=unet_noise_method,
                        number_generations_per_forward_call=number_generations_per_forward_call,
                        conv_depths=unet_depths)
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
    # the following wraps the nets above and takes care of generating the auxiliary variables at each forward call
    if continue_training_net:
        net = load_net(nets_folder + f"net{name_postfix}.pth", ConditionalGenerativeModel, inner_net,
                        size_auxiliary_variable=auxiliary_var_size, base_measure=base_measure,
                        number_generations_per_forward_call=number_generations_per_forward_call, seed=seed + 1)
    else:
        net = ConditionalGenerativeModel(inner_net, size_auxiliary_variable=auxiliary_var_size, seed=seed + 1,
                                            number_generations_per_forward_call=number_generations_per_forward_call,
                                            base_measure=base_measure)
else:
    if continue_training_net:
        net = load_net(nets_folder + f"net{name_postfix}.pth", DiscardWindowSizeDim, inner_net)
    else:
        net = DiscardWindowSizeDim(inner_net)


# --- network tools ---
if cuda:
    net.cuda()

# optimizer

optimizer_kwargs = {"weight_decay": weight_decay}  # l2 regularization
args_dict["weight_decay"] = optimizer_kwargs["weight_decay"]
optimizer = Adam(net.parameters(), lr=lr, **optimizer_kwargs)

# scheduler
scheduler_steps = 10
scheduler_gamma = scheduler_gamma
scheduler = lr_scheduler.StepLR(optimizer, scheduler_steps, gamma=scheduler_gamma, last_epoch=-1)
args_dict["scheduler_steps"] = scheduler_steps
args_dict["scheduler_gamma"] = scheduler_gamma

if method_is_gan:
    if cuda:
        critic.cuda()
    optimizer_kwargs = {}
    optimizer_c = Adam(critic.parameters(), lr=lr_c, **optimizer_kwargs)
    # dummy scheduler:
    scheduler_c = lr_scheduler.StepLR(optimizer_c, 8, gamma=1, last_epoch=-1)

string = f"Train {method} network for {model} model with lr {lr} "
if method == "SR":
    string += f"using {scoring_rule} scoring rule"
if method_is_gan:
    string += f"and critic lr {lr_c}"
print(string)

# --- train ---
start = time()
if method_is_gan:
    # load the previous losses if available:
    if continue_training_net:
        train_loss_list_g = np.load(nets_folder + f"train_loss_g{name_postfix}.npy").tolist()
        train_loss_list_c = np.load(nets_folder + f"train_loss_c{name_postfix}.npy").tolist()
    else:
        train_loss_list_g = train_loss_list_c = None
    kwargs = {}
    if method == "WGAN_GP":
        kwargs["lambda_gp"] = lambda_gp
    train_loss_list_g, train_loss_list_c = fit_adversarial(method, data_loader_train, net, critic, optimizer, scheduler,
                                                           optimizer_c, scheduler_c, epochs, cuda,
                                                           start_epoch_training=0, use_tqdm=use_tqdm,
                                                           critic_steps_every_generator_step=
                                                           critic_steps_every_generator_step,
                                                           train_loss_list_g=train_loss_list_g,
                                                           train_loss_list_c=train_loss_list_c, **kwargs)
else:
    if continue_training_net:
        train_loss_list = np.load(nets_folder + f"train_loss{name_postfix}.npy").tolist()
        val_loss_list = np.load(nets_folder + f"val_loss{name_postfix}.npy").tolist()
    else:
        train_loss_list = val_loss_list = None
    train_loss_list, val_loss_list = fit(data_loader_train, net, loss_fn, optimizer, scheduler, epochs, cuda,
                                         val_loader=data_loader_val, early_stopping=early_stopping,
                                         start_epoch_early_stopping=0 if continue_training_net else start_epoch_early_stopping,
                                         epochs_early_stopping_interval=epochs_early_stopping_interval,
                                         start_epoch_training=0, use_tqdm=use_tqdm, train_loss_list=train_loss_list,
                                         test_loss_list=val_loss_list, 
                                         prediction_length=prediction_length, scaling_mean = scaling_mean, scaling_std = scaling_std, val_mean = scaling_mean_val, val_std = scaling_val_std)
    # compute now the final validation loss achieved by the model; it is repetition from what done before but easier
    # to do this way
final_validation_loss = test_epochlongerpredictionbatch(data_loader_val, net, loss_fn, cuda, prediction_length,scaling_mean,scaling_std) ### commenting this for now.
    

training_time = time() - start
print(f"Training time: {training_time:.2f} seconds")

# # print('train_loss_list', train_loss_list)
# plt.plot(train_loss_list)
# plt.title('train_loss_list, RNN Lorenz63 w=10, l=1')
# plt.show()
# plt.close()

# # print('val_loss_list', val_loss_list)
# plt.plot(val_loss_list)
# plt.title('val_loss_list, RNN Lorenz63 w=10, l=1')
# plt.show()
# plt.close()

# prediction = net(input_data_train[:2])
# print('predicted output from the model', prediction, prediction.shape)

#print(input_data_train[:10])
#print(input_data_train[:10].shape)

def make_prediction(inner_net, inputs, prediction_length, num_simulations=ensemble_size):
    """
    a function to create l (= prediction_length) step ahead predictions given the input data

    """

    device = torch.device('cuda' if cuda else 'cpu')

    net = ConditionalGenerativeModel(inner_net, size_auxiliary_variable=auxiliary_var_size, seed=seed + 1,
                                            number_generations_per_forward_call=num_simulations,
                                            base_measure=base_measure)
    net = net.to(device)
    inputs = inputs.to(device)
    # inputs_new = inputs
    # predictions = torch.zeros((inputs.shape[0], prediction_length, data_size))

    #print(inputs.shape)
    inputs = torch.unsqueeze(inputs, 0)
    #print(inputs.shape)
    initialwindow = inputs  #  (1,10,1)
    outputs1 = net(inputs)   # 1, 7, 1
    #print(outputs1.shape)
    ensemble_length = outputs1.shape[1]

    # Cloning making list for grad 7 list of 1,10,1
    windowensemble = [initialwindow.clone() for _ in range(ensemble_length)]

    forecasts = []

    for step in range(prediction_length):
        onesteps = []

        for e in range(ensemble_length):
            eoutput = net(windowensemble[e])[:,e,:] #pick model e out

            #move ensemble window down, replace with model output
            #pop off output
            eoutput = eoutput.unsqueeze(-1)  #add dimension to line up without broadcasting
            shifted = torch.cat([windowensemble[e][:, 1:, :], eoutput], dim=1)  # 1,9,1 + 1,1,1    1,10,1
            windowensemble[e] = shifted

            onesteps.append(eoutput) 

        onesteps = torch.cat(onesteps, dim=0)

        if step == 0:
            forecasts = onesteps 
        else:
            forecasts = torch.cat((forecasts, onesteps), dim=1) 

    
    #print(forecasts.shape)


    # for i in range(prediction_length):
    #     outputs = net(inputs_new)
    #     outputs_new = outputs[:, 0, :]
    #     predictions[:, i, :] = outputs_new

    #     if i < prediction_length - 1:
    #         outputs_new = outputs_new.unsqueeze(1)          # => [batch, 1, data_size]
    #         inputs_new = torch.cat([inputs_new[:, 1:, :],   # => [batch, seq_len-1, data_size]
    #                             outputs_new],            # => [batch, 1, data_size]
    #                             dim=1)      
    return forecasts

test_predictions = make_prediction(inner_net, input_data_train[0], prediction_length) #from the starting window
print('Initial Window:', input_data_train[0])
print('Next 10 (9 prediction_length) steps', input_data_train[9])
print('predicted output from the model', test_predictions, test_predictions.shape)


#print('train_loss_list', train_loss_list)
nets_folder = "results/nets/"

np.save(nets_folder + f"train_loss{name_postfix}",train_loss_list)
np.save(nets_folder + f"val_loss{name_postfix}",val_loss_list)


#plt.plot(train_loss_list)
#plt.title('Training Loss on a Single Data Sample, lr = ' + str(lr) + ', kernel = RBF, ens = ' +str(ensemble_size)) 
#plt.title('train_loss_list, RNN Lorenz63 batch = 10, ens=7, win =10, predl=9,lr=10, kernel = RBF')
#plt.show()
#plt.close()

#print('val_loss_list', val_loss_list)
#plt.plot(val_loss_list)
#plt.title('val_loss_list, RNN Lorenz63 batch = 10, ens=7, win =10, predl=9,lr=10, kernel = RBF')
#plt.show()
#plt.close()

#print(final_validation_loss)



nets_folder = "results/nets/"
#name_postfix = "_mytrainedmodelEnergyScore"  # or something else descriptive
#name_postfix = "_mytrainedmodelkernelrbf"
# Save the trained network
save_net(nets_folder + f"net{name_postfix}.pth", net)



# def save_net(path, net):
#     """Function to save the Pytorch state_dict of a network to a file."""
#     torch.save(net.state_dict(), path)


# def load_net(path, network_class, *network_args, **network_kwargs):
#     """Function to load a network from a Pytorch state_dict, given the corresponding network_class."""
#     net = network_class(*network_args, **network_kwargs)
#     net.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
#     return net.eval()  # call the network to eval model. Needed with batch normalization and dropout layers.

# #batch_size, window_size, data_size, number_generations, size_auxiliary_variable = 4, 3, 1, 5, 1
# input_size = window_size * data_size + auxiliary_var_size
# output_size = data_size
# myfcnn = createGenerativeFCNN(input_size, output_size)()
# my_pred = myfcnn(torch.randn(batch_size, window_size, data_size), torch.randn(batch_size, number_generations, auxiliary_var_size))
# print(my_pred.shape)
# print(my_pred)



# batch_size, window_size, data_size, number_generations, size_auxiliary_variable = 4, 3, 1, 5, 1
# input_size = window_size * data_size + size_auxiliary_variable
# output_size = data_size
# myfcnn = createGenerativeFCNN(input_size, output_size)()
# my_pred = myfcnn(torch.randn(batch_size, window_size, data_size), torch.randn(batch_size, number_generations, size_auxiliary_variable))
# print(my_pred.shape)
# print(my_pred)
