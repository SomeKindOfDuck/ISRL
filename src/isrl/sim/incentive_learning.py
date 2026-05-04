"""
誘引学習(Balleine, 1992)のシミュレーションプログラム
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from isrl.model.isrl import (CvtParams, ExtParams, HstParams,
                             InteroceptiveStateModel)

HDIM = 1
CDIM = 1
ODIM = 1
EDIM = 1
ALPHA_HST = 0.05
ALPHA_CVT = 0.05
ALPHA_EXT = 0.05
TAU = 1 - 1e-6
SETPOINT = 120

def learn_incentive_value(m: InteroceptiveStateModel,
                          h_init: int,
                          trial: int,
                          session: int):
    for _ in range(session):
        m.set_current_homeostatic_point(h_init)
        covert_response = m.phi_c(0)
        m.set_current_covert_response(covert_response)
        for _ in range(trial):
            phi_c = m.phi_c(0)
            reward = m.reward_function(1)
            covert_response = m.induction_function(reward)

            delta_covert_response = covert_response - phi_c

            m.update_weight_c(delta_covert_response, 0)
            m.set_current_covert_response(covert_response)


def learn_action_value(m: InteroceptiveStateModel,
                       h_init: int,
                       trial: int,
                       session: int):

    for _ in range(session):
        m.set_current_homeostatic_point(h_init)
        covert_response = m.phi_c(1)
        m.set_current_covert_response(covert_response)

        for _ in range(trial):
            phi_o = m.phi_o(1)
            phi_c = m.phi_c(1)

            reward = m.reward_function(1.)
            covert_response = m.induction_function(reward)

            delta_reward = reward - phi_o
            delta_covert_reponse = covert_response - phi_c

            m.update_weight_o(delta_reward, 1, 1)
            m.update_weight_c(delta_covert_reponse, 1)
            m.set_current_covert_response(covert_response)


def extract_phi_across(
    m: InteroceptiveStateModel,
    h_range: tuple[float, float],
    step: float
) -> pd.DataFrame:
    s, t = h_range
    hs = np.arange(s, t, step)

    phi_c = np.zeros_like(hs, dtype=float)
    phi_o = np.zeros_like(hs, dtype=float)

    for i, h in enumerate(hs):
        m.set_current_homeostatic_point(h)
        pc = m.phi_c([1])
        m.set_current_covert_response(pc)
        po = m.phi_o([1])

        phi_c[i] = pc[0]
        phi_o[i] = po[0]

    return pd.DataFrame({
        "homeostatic_point": hs,
        "phi_c": phi_c,
        "phi_o": phi_o,
    })


def incentive_learning_experiment(hst_params: HstParams,
                                  int_params: CvtParams,
                                  ext_params: ExtParams,
                                  incentive_state: int,
                                  incentive_trial: int,
                                  incentive_session: int,
                                  instrumental_state: int,
                                  instrumental_trial: int,
                                  insturumental_session: int,
                                  model_params: dict):

    incentive_agent = InteroceptiveStateModel(hst_params, int_params, ext_params)
    control_agent = InteroceptiveStateModel(hst_params, int_params, ext_params)
    learn_incentive_value(incentive_agent,
                          incentive_state,
                          incentive_trial,
                          incentive_session)

    learn_action_value(incentive_agent,
                       instrumental_state,
                       instrumental_trial,
                       insturumental_session)

    learn_action_value(control_agent,
                       instrumental_state,
                       instrumental_trial,
                       insturumental_session)

    incentive_data = extract_phi_across(incentive_agent, (0., 240.), 1.)
    control_data = extract_phi_across(control_agent, (0., 240.), 1.)

    for name, value in model_params.items():
        incentive_data[name] = value
        control_data[name] = value

    return (incentive_data, control_data), (incentive_agent, control_agent)


def plot_preview(high2low_experimental_data: pd.DataFrame,
                 high2low_control_data: pd.DataFrame,
                 low2high_experimental_data: pd.DataFrame,
                 low2high_control_data: pd.DataFrame):

    high2low_experimental_data = high2low_experimental_data.assign(
        condition="High to mid",
        pre_feeding="Pre-feeding"
    )
    high2low_control_data = high2low_control_data.assign(
        condition="High to mid",
        pre_feeding="No pre-feeding"
    )
    low2high_experimental_data = low2high_experimental_data.assign(
        condition="Low to high",
        pre_feeding="Pre-feeding"
    )
    low2high_control_data = low2high_control_data.assign(
        condition="Low to high",
        pre_feeding="No pre-feeding"
    )

    plot_data = pd.concat([
        high2low_experimental_data,
        high2low_control_data,
        low2high_experimental_data,
        low2high_control_data,
    ], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, condition in zip(axes, ["High to mid", "Low to high"]):
        d = plot_data[plot_data["condition"] == condition]

        for pre_feeding, dd in d.groupby("pre_feeding"):
            ax.plot(
                dd["homeostatic_point"],
                dd["phi_o"],
                label=pre_feeding
            )

        ax.set_title(condition)
        ax.set_xlabel("Homeostatic point")
        ax.set_ylabel("phi_o")
        ax.legend(frameon=False)

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
    kappa = args.kappa
    rho = args.rho

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

    # 高遮断化状態でUSの摂取後に中程度の遮断化レベルで道具的条件づけの訓練した場合
    (high2low_experimental_data, high2low_control_data), (high2low_experimental_agent, high2low_control_agent) = incentive_learning_experiment(hst_params, cvt_params, ext_params, 0, 60, 5, 60, 60, 12, model_params)

    # 低遮断化状態でUSの摂取後に高程度の遮断化レベルで道具的条件づけの訓練した場合
    (low2high_experimental_data, low2high_control_data), (low2high_experimental_agent, low2high_control_agent) = incentive_learning_experiment(hst_params, cvt_params, ext_params, 120, 60, 5, 0, 60, 12, model_params)

    if preview:
        plot_preview(high2low_experimental_data, high2low_control_data,
                     low2high_experimental_data, low2high_control_data)
    else:
        out_dir = "./data/inctv"
        param_str = "-".join([f"{k}={v}" for k, v in model_params.items()])

        os.makedirs(out_dir, exist_ok=True)
        
        print(f"Running simulation with parameters = {param_str}")

        high2low_experimental_data.to_csv(
            f"{out_dir}/high2low-experimental-{param_str}.csv", index=False
        )

        high2low_control_data.to_csv(
            f"{out_dir}/high2low-control-{param_str}.csv", index=False
        )

        low2high_experimental_data.to_csv(
            f"{out_dir}/low2high-experimental-{param_str}.csv", index=False
        )

        low2high_control_data.to_csv(
            f"{out_dir}/low2high-control-{param_str}.csv", index=False
        )


if __name__ == "__main__":
    main()
