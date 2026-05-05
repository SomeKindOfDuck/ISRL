"""
目的志向行動(Colwill and Rescorla, 1985)のシミュレーションプログラム
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from isrl.model import ArrayLike
from isrl.model.isrl import (CvtParams, ExtParams, HstParams,
                             InteroceptiveStateModel)
from isrl.sim import extract_model_weights

HDIM = 2
CDIM = 2
ODIM = 2
EDIM = 1
ALPHA_HST = [0.05] * HDIM
ALPHA_CVT = [0.05] * CDIM
ALPHA_EXT = 0.05
TAU = [1 - 1e-6] * HDIM
SETPOINT = [120] * HDIM


def learn_action_value(
    m: InteroceptiveStateModel,
    trial: int,
    session: int
):
    """
    Colwill and Rescorla (1985)に従って、
    異なる道具的反応を異なる報酬で別セッションで訓練する。
    """

    # action x homeostatic dimension
    # action 0 -> homeostatic dimension 0
    # action 1 -> homeostatic dimension 1
    action2homeo = np.array([
        [1., 0.],
        [0., 1.],
    ])

    for s in range(2 * session):
        m.set_current_homeostatic_point([0, 0])

        covert_response = m.phi_c(1)
        m.set_current_covert_response(covert_response)

        if s % 2:
            rsig = np.array([1., 0.])
            action = np.array([1., 0.])
            action_index = 0
        else:
            rsig = np.array([0., 1.])
            action = np.array([0., 1.])
            action_index = 1

        for _ in range(trial):
            phi_o = m.phi_o(1)
            phi_c = m.phi_c(1)

            reward = m.reward_function(rsig)
            covert_response = m.induction_function(reward)

            homeo_mask = action2homeo[action_index]
            reward_scalar = np.sum(reward * homeo_mask)

            delta_reward = np.zeros(phi_o.size)
            delta_reward[action_index] = reward_scalar - phi_o[action_index]

            delta_covert_reponse = covert_response - phi_c

            m.update_weight_o(delta_reward, action, 1)
            m.update_weight_c(delta_covert_reponse, 1)
            m.set_current_covert_response(covert_response)


def phi_o_at_test(
    m: InteroceptiveStateModel,
    h_test: np.ndarray
) -> pd.DataFrame:
    h_baseline = np.zeros(2)
    h_test = np.asarray(h_test, dtype=float).reshape(-1)
    h_test_b = h_test[::-1]

    # Baseline
    m.set_current_homeostatic_point(h_baseline)
    covert_response = m.phi_c(1)
    m.set_current_covert_response(covert_response)
    phi_o_baseline = m.phi_o(1)

    # Test A
    m.set_current_homeostatic_point(h_test)
    covert_response = m.phi_c(1)
    m.set_current_covert_response(covert_response)
    phi_o_test_a = m.phi_o(1)

    # Test B: h_testを逆転させたもの
    m.set_current_homeostatic_point(h_test_b)
    covert_response = m.phi_c(1)
    m.set_current_covert_response(covert_response)
    phi_o_test_b = m.phi_o(1)

    return pd.DataFrame(
        {
            "Phase": [
                "Baseline", "Baseline",
                "Test A", "Test A",
                "Test B", "Test B",
            ],
            "action": [
                "Action A", "Action B",
                "Action A", "Action B",
                "Action A", "Action B",
            ],
            "homeostatic_point": [
                h_baseline[0], h_baseline[1],
                h_test[0], h_test[1],
                h_test_b[0], h_test_b[1],
            ],
            "phi_o": [
                phi_o_baseline[0], phi_o_baseline[1],
                phi_o_test_a[0], phi_o_test_a[1],
                phi_o_test_b[0], phi_o_test_b[1],
            ],
        }
    )


def goal_directed_experiment(
        hst_params: HstParams,
        cvt_params: CvtParams,
        ext_params: ExtParams,
        model_params: dict):

    agent = InteroceptiveStateModel(hst_params, cvt_params, ext_params)

    learn_action_value(agent, 120, 5)
    data = phi_o_at_test(agent, np.array([120, 0]))

    for name, value in model_params.items():
        data[name] = value

    return data, agent


def plot_preview(df):
    phases = ["Baseline", "Test A", "Test B"]
    actions = ["Action A", "Action B"]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

    for ax, phase in zip(axes, phases):
        sub = df[df["Phase"] == phase]
        sub = sub.set_index("action").loc[actions]

        y = sub["phi_o"].values

        ax.bar(actions, y, width=0.6)

        ax.set_title(phase)
        ax.set_xlabel("")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(actions, rotation=0)

    axes[0].set_ylabel("phi_o")

    plt.tight_layout()
    plt.show()


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("homeostatic_kernel", type=int)
    parser.add_argument("covert_kernel", type=int)
    parser.add_argument("--kappa", "-k", type=float, default=1.)
    parser.add_argument("--rho", "-r", type=float, default=40.)
    parser.add_argument("--preview", "-p", action="store_true")

    args = parser.parse_args()

    k_h = args.homeostatic_kernel
    k_c = args.covert_kernel
    kappa = [args.kappa] * 2
    rho = [args.rho] * 2

    hst_params = HstParams(HDIM, CDIM, ODIM, ALPHA_HST, TAU, kappa, rho, SETPOINT, k_h)
    cvt_params = CvtParams(CDIM, ODIM, ALPHA_CVT, TAU, k_c)
    ext_params = ExtParams(EDIM, CDIM, ODIM, ALPHA_EXT)

    model_params = {
        "homeostatic_kernel": args.homeostatic_kernel,
        "covert_kernel": args.covert_kernel,
        "kappa": args.kappa,
        "rho": args.rho,
    }

    return hst_params, cvt_params, ext_params, model_params, args.preview


def main():
    import os

    hst_params, cvt_params, ext_params, model_params, preview = parse_args()

    results, agent = goal_directed_experiment(hst_params, cvt_params, ext_params, model_params)

    if preview:
        plot_preview(results)
    else:
        out_dir = "./data/goal-directed"
        param_str = "-".join([f"{k}={v}" for k, v in model_params.items()])

        os.makedirs(out_dir, exist_ok=True)

        print(f"Running simulation with parameters = {param_str}")

        weights = extract_model_weights(agent, model_params)

        results.to_csv(
            f"{out_dir}/goal-directed-results-{param_str}.csv", index=False
        )
        weights.to_csv(
            f"{out_dir}/goal-directed-weights-{param_str}.csv", index=False
        )

if __name__ == "__main__":
    main()
