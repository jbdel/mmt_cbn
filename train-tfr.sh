export DATA_PATH=/home/jb/work/data/WMT17/2016/processed

export VOCAB_SOURCE=${DATA_PATH}/train.norm.tok.lc.bpe10000.vocab.en
export VOCAB_TARGET=${DATA_PATH}/train.norm.tok.lc.bpe10000.vocab.de
export TRAIN_SOURCES_TXT=${DATA_PATH}/train.norm.tok.lc.bpe10000.en
export TRAIN_SOURCES_IMG=${DATA_PATH}/train.norm.tok.lc.bpe10000.en
export TRAIN_TARGETS=${DATA_PATH}/train.norm.tok.lc.bpe10000.de
export DEV_SOURCES=${DATA_PATH}/val.norm.tok.lc.bpe10000.en
export DEV_TARGETS=${DATA_PATH}/val.norm.tok.lc.bpe10000.de
export DEV_SOURCES=${DATA_PATH}/val.norm.tok.lc.bpe10000.en


export TRAIN_SOURCES_TFRECORD0=${DATA_PATH}/dataset_resnet/dataset_train0.tfrecords
export TRAIN_SOURCES_TFRECORD1=${DATA_PATH}/dataset_resnet/dataset_train1.tfrecords
export TRAIN_SOURCES_TFRECORD2=${DATA_PATH}/dataset_resnet/dataset_train2.tfrecords
export TRAIN_SOURCES_TFRECORD3=${DATA_PATH}/dataset_resnet/dataset_train3.tfrecords
export TRAIN_SOURCES_TFRECORD4=${DATA_PATH}/dataset_resnet/dataset_train4.tfrecords
export TRAIN_SOURCES_TFRECORD5=${DATA_PATH}/dataset_resnet/dataset_train5.tfrecords
export TRAIN_SOURCES_TFRECORD6=${DATA_PATH}/dataset_resnet/dataset_train6.tfrecords
export TRAIN_SOURCES_TFRECORD7=${DATA_PATH}/dataset_resnet/dataset_train7.tfrecords
export TRAIN_SOURCES_TFRECORD8=${DATA_PATH}/dataset_resnet/dataset_train8.tfrecords
export TRAIN_SOURCES_TFRECORD9=${DATA_PATH}/dataset_resnet/dataset_train9.tfrecords
export TRAIN_SOURCES_TFRECORD10=${DATA_PATH}/dataset_resnet/dataset_train10.tfrecords
export TRAIN_SOURCES_TFRECORD11=${DATA_PATH}/dataset_resnet/dataset_train11.tfrecords
export TRAIN_SOURCES_TFRECORD12=${DATA_PATH}/dataset_resnet/dataset_train12.tfrecords

export DEV_SOURCES_TFRECORD=${DATA_PATH}/dataset_resnet/dataset_val0.tfrecords


export DEV_TARGETS_REF=${DATA_PATH}/val.norm.tok.lc.de
export TRAIN_STEPS=1000000

export MODEL_DIR=train
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./example_configs/nmt_large.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: TFRecordInputPipeline
    params:
      files:
        - $TRAIN_SOURCES_TFRECORD0
        - $TRAIN_SOURCES_TFRECORD1
        - $TRAIN_SOURCES_TFRECORD2
        - $TRAIN_SOURCES_TFRECORD3
        - $TRAIN_SOURCES_TFRECORD4
        - $TRAIN_SOURCES_TFRECORD5
        - $TRAIN_SOURCES_TFRECORD6
        - $TRAIN_SOURCES_TFRECORD7
        - $TRAIN_SOURCES_TFRECORD8
        - $TRAIN_SOURCES_TFRECORD9
        - $TRAIN_SOURCES_TFRECORD10
        - $TRAIN_SOURCES_TFRECORD11
        - $TRAIN_SOURCES_TFRECORD12" \
  --input_pipeline_dev "
    class: TFRecordInputPipeline
    params:
       files:
        - $DEV_SOURCES_TFRECORD" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR \
  --save_checkpoints_secs 999999999999999999999999999999999999

