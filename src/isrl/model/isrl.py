from typing import NamedTuple

import numpy as np

from isrl.model import ArrayLike, cvtm, extm, hstm


class HstParams(NamedTuple):
    h_dim: int
    c_dim: int
    o_dim: int
    alpha: ArrayLike
    tau: ArrayLike
    kappa: ArrayLike
    rho: ArrayLike
    setpoint: ArrayLike
    k: int

class CvtParams(NamedTuple):
    c_dim: int
    o_dim: int
    alpha: ArrayLike
    tau: ArrayLike
    k: int

class ExtParams(NamedTuple):
    e_dim: int
    c_dim: int
    o_dim: int
    alpha: ArrayLike


class InteroceptiveStateModel(object):
    def __init__(self,
                 hparams: HstParams,
                 iparams: CvtParams,
                 eparams: ExtParams,
                 ):
        self.hst = hstm.HomeostaticModule(*hparams)
        self.int = cvtm.CovertResponseModule(*iparams)
        self.ext = extm.ExteroceptiveModule(*eparams)

    def phi_c(self, exteroceptive_cue: np.ndarray | list | float) -> np.ndarray:
        return self.hst.phi_c() + self.int.phi_c() + self.ext.phi_c(exteroceptive_cue)

    def phi_o(self, exteroceptive_cue: np.ndarray | list | float) -> np.ndarray:
        return self.hst.phi_o() + self.int.phi_o() + self.ext.phi_o(exteroceptive_cue)

    def update_weight_c(self,
                        delta: ArrayLike,
                        exteroceptive_cue: ArrayLike):
        self.hst.update_weight_c(delta)
        self.ext.update_weight_c(delta, exteroceptive_cue)

    def update_weight_o(self,
                        delta: ArrayLike,
                        action: ArrayLike,
                        exteroceptive_cue: ArrayLike):
        self.hst.update_weight_o(delta, action)
        self.int.update_weight_o(delta, action)
        self.ext.update_weight_o(delta, action, exteroceptive_cue)

    def set_current_homeostatic_point(self, current_point: ArrayLike):
        self.hst.set_current_points(current_point)

    def set_current_covert_response(self, current_point: ArrayLike):
        self.int.set_current_points(current_point)

    def reward_function(self, reward_signal: ArrayLike) -> np.ndarray:
        return self.hst.reward_function(reward_signal)

    def induction_function(self, reward: ArrayLike) -> np.ndarray:
        return self.hst.induction_function(reward)
