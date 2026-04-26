def compute_pos_weight(y_train, cap=None):
    n_positive = (y_train == 1).sum()
    n_negative = (y_train == 0).sum()

    if n_positive == 0:
        raise ValueError(
            f"No positive labels found in y_train ({n_negative} negative samples, 0 positive). "
            "Check that your dataset contains anomalies and that the label column is correct."
        )

    raw_pw = float(n_negative / n_positive)
    return min(raw_pw, cap) if cap else raw_pw