# from rctorch import RcNetwork, data
# from get_transformer_training_data import x_train, y_train
# import torch
# import warnings
# warnings.filterwarnings('ignore', category=UserWarning)


# # fp_data = data.load("forced_pendulum", train_proportion = 0.2)

# # force_train, force_test = fp_data["force"]
# # target_train, target_test = fp_data["target"]

# hps = {'connectivity': 0.4,
#            'spectral_radius': 1.13,
#            'n_nodes': 202,
#            'regularization': 1.69,
#            'leaking_rate': 0.0098085,
#            'bias': 0.49}

# my_rc = RcNetwork(**hps, random_state=210, feedback=True)

# my_rc.fit(x_train, y_train)

# # # #making our prediction
# # # score, prediction = my_rc.test(y = target_test)
# # # my_rc.combined_plot()