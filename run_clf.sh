GPUDEV=${1}
TASK=${2} #sst, sicke2b
MODE=${3} #pero, pero-abl
STARTIDX=${4}
OUTDIR=${5}

NUMEPOCHS=100
TRAINSIZE=10
LMODEL="roberta-large"

if [ "$TASK" == "sst2" ]
then
    DATADIR="./data/SST-2/"
elif [ "$TASK" == "sicke2b" ]
then
    DATADIR="./data/SICK-E-balanced/2-balance"
fi

if [ "${MODE}" == "pero-abl" ]
then
    SEPMODE=" "
    DROPMATCH=" "
else
    SEPMODE="--train_global_sep"
    DROPMATCH="--drop_match"
fi

cmd="CUDA_VISIBLE_DEVICES=${GPUDEV} python main.py --restrict_dev_set --select_using_dev --sep_train_epochs 10 --balanced_dev --sep_init eos \
            --task_name ${TASK} \
            --model_name ${LMODEL} \
            --output_dir ${OUTDIR} \
            --data_dir ${DATADIR} \
            --do_train --do_eval --do_test \
            --n_select ${TRAINSIZE} \
            --select_start_index ${STARTIDX} \
            --eval_freq_steps ${NUMEPOCHS} \
            --num_train_epochs ${NUMEPOCHS} \
            --eval_batch_size 10 --train_batch_size 3 \
            ${SEPMODE} ${DROPMATCH}"
echo "${cmd}"
eval ${cmd}
