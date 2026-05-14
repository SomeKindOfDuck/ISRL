"""
このプログラムはISRLのうちUSやCSによって誘発された潜在的反応を手がかりとした学習を担うモジュールを提供する
"""

import numpy as np

from isrl.model import ArrayLike, gaussian_kernel


class CovertResponseModule(object):
    """
    このクラスは潜在的反応を手がかりとして表現し学習を担う。
    """
    def __init__(
            self,
            c_dim: int,
            o_dim: int,
            alpha: ArrayLike,
            tau: ArrayLike,
            k: int = 10,
    ):
        """

        :c_dim: int: 潜在的反応空間の次元数
        :o_dim: int: 顕現的反応空間の次元数
        :alpha : np.ndarray | list | float: 潜在的反応ごとの学習率
        :tau: np.ndarray | list | float: 誘発された反応が減衰する速度
        :k: int: 内部反応空間を強化学習の状態として近似する際のカーネル数（ガウシアン）

        """

        alpha = np.asarray(alpha, dtype=float).reshape(-1)
        tau = np.asarray(tau, dtype=float).reshape(-1)

        if alpha.size != c_dim or tau.size != c_dim:
            raise ValueError(
                f"alpha, tau must have length c_dim={c_dim}. "
                f"Got alpha={alpha.size}, tau={tau.size}."
            )

        self._c_dim = c_dim
        self._o_dim = o_dim
        self._current_point = np.zeros(c_dim)
        self._alpha = alpha
        self._tau = tau

        self._k = k
        self._centers = np.linspace(
            np.ones(c_dim, dtype=float),
            np.full(c_dim, -1, dtype=float),
            k,
            axis=0
        ).T
        self._sigma = np.mean(np.abs(np.diff(self._centers, axis=1)), axis=1)
        self._kernel = self.update_kernel()

        self._c_weights = np.zeros((self._c_dim, self._k, self._c_dim), dtype=float)
        self._o_weights = np.zeros((self._c_dim, self._k, self._o_dim), dtype=float)

    @property
    def current_point(self):
        return self._current_point

    @property
    def kernel(self):
        return self._kernel

    def update_kernel(self) -> np.ndarray:
        """ガウスカーネルによってホメオスタシス状態を強化学習の状態に近似

        returns: np.ndarray: 各次元ごとのカーネルの活性強度

        """
        return gaussian_kernel(self._current_point, self._centers, self._sigma)

    def set_current_points(
            self,
            current_point: ArrayLike,
            overwrte: bool = True
    ):
        """与えられた引数を現在のホメオスタシス状態として上書きする

        current_point: np.ndarray | list | float: 各次元のホメオスタシス状態

        """
        current_point = np.asarray(current_point, dtype=float).reshape(-1)
        if len(current_point) != self._c_dim:
            raise ValueError(f"current points must have length h_dim={self._c_dim}.")
        if overwrte:
            self._current_point = current_point
        else:
            self._current_point = (1 - self._tau) * current_point + self._current_point * self._tau
        self._kernel = self.update_kernel()

    def phi_c(self) -> np.ndarray:
        """現在のホメオスタシス状態での潜在的反応ごとのポテンシャル

        :returns: 潜在的反応ポテンシャル

        """
        # (h_dim,k) × (h_dim,k,c_dim) -> (c_dim,)
        return np.einsum("hk,hkc->c", self.kernel, self._c_weights)

    def phi_o(self) -> np.ndarray:
        """現在のホメオスタシス状態での外的反応ごとのポテンシャル

        :returns: 顕現的反応ポテンシャル

        """
        # (h_dim,k) × (h_dim,k,o_dim) -> (o_dim,)
        return np.einsum("hk,hko->o", self.kernel, self._o_weights)

    # def update_weight_o(self, delta: ArrayLike, action: ArrayLike):
    #     """顕現的反応の重み更新

    #     :delta: 顕現反応によって得られた報酬の予測誤差
    #     :action: 行動のone-hot vector

    #     """
    #     delta = np.asarray(delta, dtype=float).reshape(-1)
    #     action = np.asarray(action, dtype=float).reshape(-1)
    #     delta = delta * action
    #     self._o_weights += self._alpha * np.einsum("h,hk->kh", delta, self.kernel)
    def update_weight_o(self, delta: ArrayLike, action: ArrayLike):
        delta = np.asarray(delta, dtype=float).reshape(-1)
        action = np.asarray(action, dtype=float).reshape(-1)

        if action.size != self._o_dim:
            raise ValueError(f"action must have length o_dim={self._o_dim}.")

        if delta.size != self._o_dim:
            raise ValueError(f"delta must have length o_dim={self._o_dim}.")

        delta_o = delta * action

        self._o_weights += np.einsum(
            "c,ck,o->cko",
            self._alpha,
            self.kernel,
            delta_o
        )

    def update_weight_c(self, delta: ArrayLike):
        """潜在的反応の重み更新

        :delta: 現在の潜在的反応と誘導された反応の誤差

        """
        delta = np.asarray(delta, dtype=float).reshape(-1)
        self._c_weights += self._alpha * np.einsum("h,hk->kh", delta, self.kernel)
