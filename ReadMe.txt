

Pipeline:

A] Preprocess

1. python pick_subset.py
2. python build_answer_vocab.py
3. python build_layouts.py
4. python build_module_input_vocab.py
5. python extract_image_vgg_features.py

B] Models:

1. Composable NNs (Main Work):
   python train_test_cmp_nn_vqa.py
2. LSTM+IMG Baseline:
   python train_test_baseline_lstm_img.py
3. LSTM Baseline:
   python train_test_baseline_lstm.py


C] Demo:
   
   ... todo ...

and be happy! : )
