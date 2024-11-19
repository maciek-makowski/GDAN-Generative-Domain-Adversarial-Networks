#! /bin/bash
# Job Name
#$ -N Perdomo_data_experiment 
# Notifications
#$ -<your_email_address@abc.com>
# When notified (b : begin, e : end, s : error)
#$ -m es
# Execute from the current directory, and pass environment:
#$ -cwd # use current directory
#$ -V  # pass my complete environment to the job
# define output file name
#$ -o ./output.o
# define error file name
#$ -e ./error_output.o
#

echo 'Start of Job:'

eval "$(/nethome/1177729/anaconda3/bin/conda shell.bash hook)"

conda activate myenv_1177729

echo 'Environment activated.'

python Perdomo_data.s
#python Izzo_binary_classification.py

echo 'Finished computation.'

