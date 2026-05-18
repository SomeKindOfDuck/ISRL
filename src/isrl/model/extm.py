"""
このプログラムはISRLのうち、外受容感覚を手がかりとした学習を担うモジュールを提供する
"""

import numpy as np

from isrl.model import ArrayLike


class ExteroceptiveModule(object):
    """
    このクラスは外部環境に基づく価値学習に使用する
    """

    def __init__(
            self,
            e_dim: int,
            c_dim: int,
            o_dim: int,
            alpha: ArrayLike,
    ):
        """
        TODO: 引数の説明
        """

        alpha = np.asarray(alpha, dtype=float).reshape(-1)

        if not isinstance(e_dim, int) or e_dim <= 0:
            raise ValueError("h_dim must be a positive integer.")

        if alpha.size != e_dim:
            raise ValueError(
                f"alpha must have length e_dim={e_dim}. "
                f"Got alpha={alpha.size}."
            )

        self._e_dim = e_dim
        self._c_dim = c_dim
        self._o_dim = o_dim
        self._alpha = alpha

        # 以下はh→c,oの結合重み
        # c, oはそれぞれcovert, overt responseで内的・外的行動を意味する
        # h_dim×k から c_dim, o_dimが出力になるようにそれぞれの重みを0で初期化
        # 最終的にhomeostatic_cue * weightsでそれぞれの強度が計算できるようにする
        self._c_weights = np.zeros((self._e_dim, self._c_dim), dtype=float)
        self._o_weights = np.zeros((self._e_dim, self._o_dim), dtype=float)

    def phi_c(self, exteroceptive_cue: ArrayLike) -> np.ndarray:
        """現在の観測下の潜在的反応ごとのポテンシャル

        :returns: 潜在的反応ポテンシャル

        """
        # (h_dim,k) × (h_dim,k,c_dim) -> (c_dim,)
        exteroceptive_cue = np.asarray(exteroceptive_cue, dtype=float).reshape(-1)
        return np.einsum("e,ec->c", exteroceptive_cue, self._c_weights)

    def phi_o(self, exteroceptive_cue: ArrayLike) -> np.ndarray:
        """現在の観測下での外的反応ごとのポテンシャル

        :returns: 顕現的反応ポテンシャル

        """
        # (h_dim,k) × (h_dim,k,o_dim) -> (o_dim,)
        exteroceptive_cue = np.asarray(exteroceptive_cue, dtype=float).reshape(-1)
        return np.einsum("e,eo->o", exteroceptive_cue, self._o_weights)

    def update_weight_o(
        self,
        delta: ArrayLike,
        action: ArrayLike,
        exteroceptive_cue: ArrayLike,
    ):
        delta = np.asarray(delta, dtype=float).reshape(-1)
        action = np.asarray(action, dtype=float).reshape(-1)
        exteroceptive_cue = np.asarray(exteroceptive_cue, dtype=float).reshape(-1)

        if exteroceptive_cue.size != self._e_dim:
            raise ValueError(
                f"exteroceptive_cue must have length e_dim={self._e_dim}. "
                f"Got {exteroceptive_cue.size}."
            )

        if action.size != self._o_dim:
            raise ValueError(
                f"action must have length o_dim={self._o_dim}. "
                f"Got {action.size}."
            )

        if delta.size != self._o_dim:
            raise ValueError(
                f"delta must have length o_dim={self._o_dim}. "
                f"Got {delta.size}."
            )

        delta_o = delta * action

        self._o_weights += np.einsum(
            "e,e,o->eo",
            self._alpha,
            exteroceptive_cue,
            delta_o
        )

    def update_weight_c(
        self,
        delta: ArrayLike,
        exteroceptive_cue: ArrayLike
    ):
        """潜在的反応の重み更新

        :delta: 顕現反応によって得られた報酬の予測誤差
        :action: 行動のone-hot vector

        """
        delta = np.asarray(delta, dtype=float).reshape(-1)
        exteroceptive_cue = np.asarray(exteroceptive_cue, dtype=float).reshape(-1)
        self._c_weights += self._alpha * np.einsum("e,c->ec", exteroceptive_cue, delta)
