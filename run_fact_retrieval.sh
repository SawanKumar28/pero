GPUDEV=${1}
MODE=${2} #pero, pero-abl
STARTIDX=${3}
OUTDIR=${4}

NUMEPOCHS=30
TRAINSIZE=10
LMODEL=" bert-large-cased"

DATADIR="./data/fact-retrieval/original/"

if [ "${MODE}" == "pero-abl" ]
then
    SEPMODE=" "
    DROPMATCH=" "
else
    SEPMODE="--train_global_sep"
    DROPMATCH="--drop_match"
fi

all_relations="P1001 P101 P103 P106 P108 P127 P1303 P131 P136 P1376 P138 P140 P1412 P159 P17 P176 P178 P19 P190 P20 P264 P27 P276 P279 P30 P31 P36 P361 P364 P37 P39 P407 P413 P449 P463 P47 P495 P527 P530 P740 P937"
for rel in ${all_relations};
do
    echo "Relation ""${rel}"
    DATADIR="./data/fact-retrieval/original/"
    TASK="fact-retrieval_R_""${rel}"
    OUTPUTDIR="${OUTDIR}/r${rel}"
    cmd="CUDA_VISIBLE_DEVICES=${GPUDEV} python main.py --restrict_dev_set --select_using_dev --sep_train_epochs 5 --balanced_dev --sep_init eos \
            --task_name ${TASK} \
            --model_name ${LMODEL} \
            --output_dir ${OUTPUTDIR} \
            --data_dir ${DATADIR} \
            --do_train --do_eval --do_test \
            --n_select ${TRAINSIZE} \
            --select_start_index ${STARTIDX} \
            --eval_freq_steps ${NUMEPOCHS} \
            --num_train_epochs ${NUMEPOCHS} \
            --eval_batch_size 10  \
            ${SEPMODE} ${DROPMATCH}"
    echo "${cmd}"
    eval ${cmd}
done
