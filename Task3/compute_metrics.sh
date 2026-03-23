
# python evaluate_offline.py --preds_file /ghome/group05/gerard/MCV-C5/Task3/predictions_recovered/predictions_baseline-resnet18-gru-char.json \
#                         --epoch 10 \
#                         --output_dir metrics_computed \
#                         --run_name baseline-resnet18-gru-char

# python evaluate_offline.py --preds_file /ghome/group05/gerard/MCV-C5/Task3/predictions_recovered/predictions_exp-resnet50-gru-char.json \
#                         --epoch 10 \
#                         --output_dir metrics_computed \
#                         --run_name exp-resnet50-gru-char

# python evaluate_offline.py --preds_file /ghome/group05/gerard/MCV-C5/Task3/predictions_recovered/predictions_exp-resnet50-gru-subword_lr1e-4_bs64_maxlen50.json \
#                         --epoch 10 \
#                         --output_dir metrics_computed \
#                         --run_name exp-resnet50-gru-subword_lr1e-4_bs64_maxlen50

# python evaluate_offline.py --preds_file /ghome/group05/gerard/MCV-C5/Task3/predictions_recovered/predictions_exp-resnet50-gru-word_lr1e-4_bs64_maxlen50.json \
#                         --epoch 10 \
#                         --output_dir metrics_computed \
#                         --run_name exp-resnet50-gru-word_lr1e-4_bs64_maxlen50

# python evaluate_offline.py --preds_file /ghome/group05/gerard/MCV-C5/Task3/predictions_recovered/predictions_exp-resnet50-lstm_2layers-word_lr1e-4_bs64_maxlen50.json \
#                         --epoch 10 \
#                         --output_dir metrics_computed \
#                         --run_name exp-resnet50-lstm_2layers-word_lr1e-4_bs64_maxlen50

# python evaluate_offline.py --preds_file /ghome/group05/gerard/MCV-C5/Task3/predictions_recovered/predictions_exp-resnet50-gru-word_lr1e-4_bs64_maxlen50_teacher_forcing.json \
#                         --epoch 10 \
#                         --output_dir metrics_computed \
#                         --run_name exp-resnet50-gru-word_lr1e-4_bs64_maxlen50_teacher_forcing

# python evaluate_offline.py --preds_file /ghome/group05/gerard/MCV-C5/Task3/predictions_recovered/predictions_exp-resnet50-lstm_2layers-word_lr1e-4_bs64_maxlen50_teacher_forcing.json \
#                         --epoch 10 \
#                         --output_dir metrics_computed \
#                         --run_name exp-resnet50-lstm_2layers-word_lr1e-4_bs64_maxlen50_teacher_forcing