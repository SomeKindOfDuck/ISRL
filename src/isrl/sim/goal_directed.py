
"""
目的志向行動(Colwill and Rescorla, 1985)のシミュレーションプログラム
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from isrl.model.isrl import (CvtParams, ExtParams, HstParams,
                             InteroceptiveStateModel)

HDIM = 2
CDIM = 2
ODIM = 2
EDIM = 1
ALPHA_HST = [0.05] * HDIM
ALPHA_CVT = [0.05] * CDIM
ALPHA_EXT = 0.05
TAU = [1 - 1e-6] * HDIM
SETPOINT = [120] * HDIM

hst_params = HstParams(HDIM, CDIM, ODIM, ALPHA_HST, TAU, [1.] * HDIM, [40.] * HDIM, SETPOINT, 5)
cvt_params = CvtParams(CDIM, ODIM, ALPHA_CVT, TAU, 7)
ext_params = ExtParams(EDIM, CDIM, ODIM, ALPHA_EXT)

m = InteroceptiveStateModel(hst_params, cvt_params, ext_params)


def learn_action_value(m: InteroceptiveStateModel,
                       trial: int,
                       session: int):
    """
    Colwill and Rescorla (1985)に従って、異なる道具的反応を異なる報酬で別セッションで訓練する。
    """

    for s in range(2 * session):
        m.set_current_homeostatic_point([0, 0])
        covert_response = m.phi_c(1)
        m.set_current_covert_response(covert_response)
        if s % 2:
            rsig = np.array([1., 0.])
            action = np.array([1, 0])
        else:
            rsig = np.array([0., 1.])
            action = np.array([0, 1])

        for _ in range(120):
            phi_o = m.phi_o(1)
            phi_c = m.phi_c(1)

            reward = m.reward_function(rsig)
            covert_response = m.induction_function(reward)

            delta_reward = np.zeros(phi_o.size)
            delta_reward = (reward - phi_o) * rsig
            delta_covert_reponse = covert_response - phi_c

            m.update_weight_o(delta_reward, action, 1)
            m.update_weight_c(delta_covert_reponse, 1)
            m.set_current_covert_response(covert_response)
