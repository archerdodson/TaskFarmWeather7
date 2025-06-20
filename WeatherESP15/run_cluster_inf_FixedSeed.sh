#!/bin/bash

#SBATCH --job-name=L1_fixed_seed
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4571
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=francesca.basini.1@warwick.ac.uk

MY_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=$MY_NUM_THREADS


module purge

module load GCC/12.2.0 
module load OpenMPI/4.1.4
module load SciPy-bundle/2023.02

source ~/Fran_PopGen/bin/activate

# ---- Run Inference for Simulated Experiment

# Running the simulated inference script
echo "Running inference FIXED SEED (Multiexperiment)"
echo ""

L=1
echo "Inference for Simulated Experiment L= $L"

# Later set hyper for which experimentø

# default_result_folder = {"Simulated": "/Simulated/",
#                         "Simulate_x0_unknown": "/Simulate_x0_unknown/",
#                         "HorseData": "/HorseData/",
#                         "Drosofila": "/Drosofila/"} 

#EXP_TYPE = ( Simulated SimulatedNoCum HorseData Drosophila )
#EXP_TYPE = ( Simulated )

# loop over LOCI, CASES and SCENARIOS

#LOCI = ( 1 2 3 )

# ---- Run Inference for Simulated Experiment
if [[ "$L" == 2 ]]; then
    SCENARIOS=( 1 2 3 4 ) 
    R_CASES=( 1 2 3 4 5 )
elif [[ "$L" == 3 ]]; then
    SCENARIOS=( 1 2) 
    R_CASES=( 1 2 3 4 5)
fi

if [[ "$L" == 1 ]]; then
    SCENARIOS=( 1 2 3 4 ) 
    Rcase=1
    for ((s=0;s<${#SCENARIOS[@]};++s)); do
        S=${SCENARIOS[s]}
        echo "Computing scenario $S"
        srun python3 0_L_all_FixedSeed_MultiExp_Inference.py Simulated --loci "${L}" \
                                            --freq_setup --scenario "${S}" --case "${Rcase}"\
                                            --multiexp 15 \
                                            --FixedSeed \
                                            --allele_data    \
                                            --n_samples 1000 \
                                            --steps 6\
                                            --sigkernel_sigma 2.5
    done
else
    for ((s=0;s<${#SCENARIOS[@]};++s)); do
        S=${SCENARIOS[s]}
        for ((r=0;r<${#R_CASES[@]};++r)); do
            R=${R_CASES[r]}
            echo "Computing scenario $S and R Case $R"
            srun python3 0_L_all_FixedSeed_MultiExp_Inference.py Simulated --loci "${L}" \
                                                --freq_setup \
                                                --multiexp 15 \
                                                --FixedSeed \
                                                --scenario "${S}" --case "${R}"\
                                                --allele_data    \
                                                --n_samples 1000 \
                                                --steps 6\
                                                --sigkernel_sigma 2.5 \
                                                --tempering \
                                                --temper_w 0.02
        done
    done
fi

    #  model --loci'
    #--scenario', type=int, default=None)
    #--case', type=int, default=None)
    #--sigkernel', type=str, default='standard')
    #--sigkernel_sigma', type=float, default=2.5)
    #--allele_data', action="store_true", 
    #--reps', type=int, default=10, 
    #--seed', type=int, default = 19, help='rng')
    #--popsize', type=int, default = 500, help='N')
    #--generations', type=int, default = 100, help='')
    #--gen_int', type=int, default = 10, help='')
    #--n_samples', type=int, default=300)
    #--n_samples_per_param', type=int, default=10)
    #--steps', type=int, default=4)                                            
    

deactivate



# for ((k2=0;k2<${#N_SAMPLES_IN_OBS[@]};++k2)); do

#     runcommand="python scripts/inference.py \
#     $model  \
#     $method  \
#     --n_samples $n_samples  \
#     --burnin $burnin  \
#     --n_samples_per_param $n_samples_per_param \
#     --n_samples_in_obs $n_samples_in_obs \
#     --inference_folder $inference_folder \
#     --observation_folder $observation_folder \
#     --sigma 52.37 \
#     --prop_size $PROPSIZE \
#     --load \
#     --n_group $NGROUP "





