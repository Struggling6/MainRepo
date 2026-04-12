
def compute_pos_weight(y_train, cap=None):
    raw_pw = float((y_train == 0).sum() / (y_train == 1).sum())
    return min(raw_pw, cap) if cap else raw_pw