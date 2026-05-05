import pandas as pd

from isrl.model.isrl import InteroceptiveStateModel


def extract_model_weights(
    m: InteroceptiveStateModel,
    model_params: dict | None = None
) -> pd.DataFrame:
    rows = []

    def add_kernel_weights(module_name, weight_type, weights, cue_type):
        for cue_dim in range(weights.shape[0]):
            for kernel_index in range(weights.shape[1]):
                for target_dim in range(weights.shape[2]):
                    rows.append({
                        "module": module_name,
                        "weight_type": weight_type,
                        "cue_type": cue_type,
                        "cue_dim": int(cue_dim),
                        "kernel_index": int(kernel_index),
                        "target_dim": int(target_dim),
                        "weight": float(weights[cue_dim, kernel_index, target_dim]),
                    })

    def add_exteroceptive_weights(module_name, weight_type, weights):
        for cue_dim in range(weights.shape[0]):
            for target_dim in range(weights.shape[1]):
                rows.append({
                    "module": module_name,
                    "weight_type": weight_type,
                    "cue_type": "exteroceptive",
                    "cue_dim": int(cue_dim),
                    "kernel_index": -1,
                    "target_dim": int(target_dim),
                    "weight": float(weights[cue_dim, target_dim]),
                })

    add_kernel_weights(
        "Homeostatic",
        "covert",
        m.hst._c_weights,
        "homeostatic",
    )
    add_kernel_weights(
        "Homeostatic",
        "overt",
        m.hst._o_weights,
        "homeostatic",
    )
    add_kernel_weights(
        "Covert",
        "covert",
        m.int._c_weights,
        "covert",
    )
    add_kernel_weights(
        "Covert",
        "overt",
        m.int._o_weights,
        "covert",
    )
    add_exteroceptive_weights(
        "Exteroceptive",
        "covert",
        m.ext._c_weights,
    )
    add_exteroceptive_weights(
        "Exteroceptive",
        "overt",
        m.ext._o_weights,
    )

    data = pd.DataFrame(rows)

    if model_params is not None:
        for name, value in model_params.items():
            data[name] = value

    return data
