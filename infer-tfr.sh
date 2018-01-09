export VOCAB_PATH=/home/jb/work/data/WMT17/2016/processed


export DATA_PATH=/home/jb/work/data/WMT17/2016/processed
export MODE=test2016


#export DATA_PATH=/home/jb/work/data/WMT17/coco
#export MODE=testcoco

#export DATA_PATH=/home/jb/work/data/WMT17/test2017
#export MODE=test2017

export MODEL_DIR=train
export PRED_DIR=${MODEL_DIR}/pred





export SOURCES_TFRECORD=${DATA_PATH}/dataset_resnet/dataset_${MODE}0.tfrecords

export TARGETS_REF=${DATA_PATH}/${MODE}.norm.tok.lc.de
export VOCAB_SOURCE=${VOCAB_PATH}/train.norm.tok.lc.bpe10000.vocab.en
export VOCAB_TARGET=${VOCAB_PATH}/train.norm.tok.lc.bpe10000.vocab.de


mkdir -p ${PRED_DIR}

python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --checkpoint_path $MODEL_DIR/model.ckpt-43000 \
  --model_params "
    vocab_source: $VOCAB_SOURCE
    vocab_target: $VOCAB_TARGET
    inference.beam_search.beam_width: 12" \
  --input_pipeline "
    class: TFRecordInputPipeline
    params:
       files:
        - $SOURCES_TFRECORD" \
  >  ${PRED_DIR}/predictions-bpe.txt


sed -r 's/(@@ )|(@@ ?$)//g' ${PRED_DIR}/predictions-bpe.txt > ${PRED_DIR}/predictions.txt

./bin/tools/multi-bleu.perl ${TARGETS_REF} < ${PRED_DIR}/predictions.txt


cd multeval-0.5.1

./multeval.sh eval --refs ${MODE}.norm.tok.lc.de --hyps-baseline ../${PRED_DIR}/predictions.txt --meteor.language de


