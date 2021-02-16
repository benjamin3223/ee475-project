def on_off_balance(np_network_input, target_df_preproc, app_seq_win, balance_target=0.75, on_watts_threshold=10):
    """ np_network_input, target_df_preproc, app_seq_win, are all numpy arrays
        target is my case a single point therefore shape is (x, 1) for the array
        app_seq_win is the appliance (target) sequence window/
    """
    if balance_target == 0.00:
        return np_network_input, target_df_preproc, app_seq_win
    balance_percentage_threshold = 0.50
    idx_on, _ = np.where(target_df_preproc >= on_watts_threshold)
    if sum(idx_on) < 1:
        return None, None, None
    percentage_on = idx_on.shape[0] / target_df_preproc.shape[0]
    print(f"Percentage On: {percentage_on}")
    if percentage_on < balance_percentage_threshold:  # Only adjust balance if less than 50% are activations
        sub_sample_target_size = int((idx_on.shape[0] / balance_target) - idx_on.shape[0])
        below_threshold = np.where(target_df_preproc <= on_watts_threshold)[0]
        sub_sample_size = below_threshold.shape[0]
        if sub_sample_size > sub_sample_target_size:
            mask = np.random.choice(below_threshold, sub_sample_target_size, replace=False)  # idx_on and below_threshold are imbalanced
        else:
            mask = np.random.choice(below_threshold, idx_on.shape[0], replace=False)  # idx_on and below_threshold are the same size e.g. 50%
        # Insort combines and sorts these for index selection
        balanced_index = research_shared.insort(idx_on, mask)
        # target_df and the network input is then reduced to balanced ON/OFF events
        target_df_preproc = target_df_preproc[balanced_index, :]
        app_seq_win = app_seq_win[balanced_index, :]
        np_network_input = np_network_input[balanced_index, :]
    return np_network_input, target_df_preproc, app_seq_win