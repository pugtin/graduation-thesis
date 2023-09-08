group = gcc50557

run:
    qsub -g $(group) run.sh $(dataset) $(train_mode) $(encoding)
run-preprocessing:
    qsub -g $(group) run-preprocessing.sh $(dataset) $(encoding)