seed_nbr = 0
chunk_len = 50
dt = 0.01
test_ratio = 0.01
val_ratio = 0.15
#How many values to take as input before predicting the next one, (should be one less than chunk_len)
inp_seq_len = chunk_len - 1
