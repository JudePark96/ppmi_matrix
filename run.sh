python3 main.py --v2i_path ./vocab/kp20k_vocab2idx.pkl \
                --i2v_path ./vocab/kp20k_idx2vocab.pkl \
                --training_corpus ./corpus/processed_kp20k_training_context_filtered_RmKeysAllUnk.txt \
                --output_path ./co_occurence_matrix.pkl \
                --windows 3