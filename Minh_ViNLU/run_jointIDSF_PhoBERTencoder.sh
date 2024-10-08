#As we initialize JointIDSF from JointBERT, user need to train a base model JointBERT first
./run_jointBERT-CRF_PhoBERTencoder.sh
#Train JointIDSF
export lr=5e-5
export c=0.15
export s=100
echo "${lr}"
export MODEL_DIR=JointIDSF_PhoBERTencoder
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$c"/"$s
echo "${MODEL_DIR}"
python3 main.py --token_level word-level \
                  --model_type phobert \
                  --model_dir $MODEL_DIR \
                  --data_dir PhoATIS \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs 20 \
                  --tuning_metric mean_intent_slot \
                  --use_intent_context_attention \
                  --attention_embedding_size 256 \
                  --use_crf \
                  --gpu_id 0 \
                  --embedding_type soft \
                  --intent_loss_coef $c \
                  --pretrained \
                  --pretrained_path JointBERT-CRF_PhoBERTencoder/3e-5/0.6/100 \
                  --learning_rate $lr \
                  --train_batch_size 32 \
                  --attention_mechanism mhsa \
                  --optimizer adam \
                  --no_cuda