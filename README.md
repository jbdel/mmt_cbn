If you just interested by the code, see below

**Reproduce experiments :**

0) Download multeval [here](https://github.com/jhclark/multeval) and place it at the root. (with name multeval-0.5.1)

1) Download pretrained resnet [from here](https://github.com/tensorflow/models/tree/master/research/slim) and put it on models folder

2) Follow data preprocessing as described [here](https://github.com/lium-lst/wmt17-mmt) (preparation subsection) and put it in a folder

3) Download raw image of Multi30K and put it in a folder 

4) Launch scripts seq2seq/data/ToTfRecord.py to create TfRecords files. Beforehand, change the dir_img and dir_txt variable from 2) and 3)

5) Start training by launching train-tfr.sh. Beforehand, change the export variables (like DATA_PATH) according to newly created tf records files and vocabulary files.

The models will be saved in models folder (create it beforehand).
Every 1000 steps, checkpoint is created and inference is run. Best meteor scores are saved in models/best-meteor.txt with its global step.
Inference scores during training are lower than real inference because it uses the "feed previous" (not real inference).

Infer-tfr.sh is true inference, see code inside (similar to test-tfr.sh)

**Code**:

-Encoder is taking place in seq2seq/encoders/rnn_encoder class BidirectionalRNNEncoder where it call the resnet.

-Resnet implementation is in seq2seq/contrib/resnet/resnet_v1.py. Method resnet_v1 uses layers.batchnorm.

-Layers class is found at seq2seq/contrib/layers.py. Method batch_norm implements conditional batch norm (line 833). You can exclude blocks from CBN at line 827 with variable exclude_scopes. You can exclude blocks from training in resnet_utils.py line 184.

-Models settings are in example_configs/nmt_large.yml.







