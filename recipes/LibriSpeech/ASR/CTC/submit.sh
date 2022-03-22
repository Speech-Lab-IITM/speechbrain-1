#! /bin/bash
#SBATCH --nodes=1
#SBATCH --partition=nltmp
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=160:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"
export ftp_proxy=http://172.50.0.50:9090
export https_proxy=http://172.50.0.50:9090
export http_proxy=http://172.50.0.50:9090
source /nlsasfs/home/nltm-pilot/arunk/speechbrain-1/env_speechbrain/bin/activate
python train.py hparams/transformer_wav2vec2.yaml --tag "weightedsum_2layer_bs32_ag4"
