"""
このプログラムはISRLのうち、ホメオスタシス状態に基づく報酬の計算と、状態を手がかりとした学習を担うモジュールを提供する

"""

import numpy as np

from isrl.model import ArrayLike, gaussian_kernel


class HomeostaticModule(object):
    """
    このクラスはホメオスタシス空間の定義・変化、及び報酬の計算を行う
    """

    def __init__(
            self,
            h_dim: int,
            c_dim: int,
            o_dim: int,
            alpha: ArrayLike,
            tau: ArrayLike,
            kappa: ArrayLike,
            rho: ArrayLike,
            setpoint: ArrayLike | None = None,
            k: int = 10,
    ):
        """

        :h_dim: int: ホメオスタシス空間の次元数
        :c_dim: int: 潜在的反応の次元数（生理反応など観察できない身体内部の反応）
        :o_dim: int: 顕現的反応の次元数（骨格筋運動を伴う観察可能な行動）
        :alpha : np.ndarray | list | float: ホメオスタシス次元ごとの学習率
        :tau: np.ndarray | list | float: 各次元ごとのホメオスタシス状態の時定数
        :kappa: np.ndarray | list | float: ホメオスタシス状態の変化に対する報酬価値の変化しやすさ
        :rho: np.ndarray | list | float: 報酬価値が減少し始めるところ
        :setpoint: np.ndarray | list | float: 各次元ごとのセットポイント（理想状態）
        :k: int: ホメオスタシス空間を強化学習の状態として近似する際のカーネル数（ガウシアン）

        """

        alpha = np.asarray(alpha, dtype=float).reshape(-1)
        tau = np.asarray(tau, dtype=float).reshape(-1)
        rho = np.asarray(rho, dtype=float).reshape(-1)
        kappa = np.asarray(kappa, dtype=float).reshape(-1)
        if setpoint is None:
            setpoint = np.zeros(h_dim)
        setpoint = np.asarray(setpoint, dtype=float).reshape(-1)

        if not isinstance(h_dim, int) or h_dim <= 0:
            raise ValueError("h_dim must be a positive integer.")

        if alpha.size != h_dim or tau.size != h_dim or rho.size != h_dim or kappa.size != h_dim or setpoint.size != h_dim:
            raise ValueError(
                f"alpha, tau, m, n, setpoint must have length h_dim={h_dim}. "
                f"Got tau={tau.size}, m={rho.size}, n={kappa.size}, setpoint={setpoint.size}."
            )

        self._h_dim = h_dim
        self._c_dim = c_dim
        self._o_dim = o_dim
        self._setpoint = setpoint
        self._current_point = np.zeros(h_dim)
        self._previous_point = np.zeros(h_dim)
        self._alpha = alpha
        self._tau = tau
        self._kappa = kappa
        self._rho = rho

        # 以下はgaussian kernelによるhomeostatic cueの近似
        self._k = k
        self._centers = np.linspace(
            np.zeros(h_dim, dtype=float),
            self._setpoint * 2.0,
            k,
            axis=0
        ).T
        self._sigma = np.mean(np.abs(np.diff(self._centers, axis=1)), axis=1)
        self._kernel = self.update_kernel()

        # 以下はh→c,oの結合重み
        # c, oはそれぞれcovert, overt responseで内的・外的行動を意味する
        # h_dim×k から c_dim, o_dimが出力になるようにそれぞれの重みを0で初期化
        # 最終的にhomeostatic_cue * weightsでそれぞれの強度が計算できるようにする
        self._c_weights = np.zeros((self._h_dim, self._k, self._c_dim), dtype=float)
        self._o_weights = np.zeros((self._h_dim, self._k, self._o_dim), dtype=float)

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

    def set_setpoint(
            self,
            setpoint: ArrayLike
    ):
        """与えられた引数を新たなセットポイントとして上書きする

        :setpoint: np.ndarray | list | float: 各次元のセットポイント

        """
        setpoint = np.asarray(setpoint, dtype=float).reshape(-1)
        if setpoint.size != self._h_dim:
            raise ValueError(f"setpoints must have length h_dim={self._h_dim}.")
        self._setpoint = setpoint
        self._centers = np.linspace(
            np.zeros(self._h_dim, dtype=float),
            self._setpoint * 2.0,
            self._k,
            axis=0
        ).T
        self._sigma = np.mean(np.abs(np.diff(self._centers, axis=1)), axis=1)
        self._kernel = self.update_kernel()

    def set_current_points(
            self,
            current_point: ArrayLike
    ):
        """与えられた引数を現在のホメオスタシス状態として上書きする

        current_point: np.ndarray | list | float: 各次元のホメオスタシス状態

        """
        current_point = np.asarray(current_point, dtype=float).reshape(-1)
        if len(current_point) != self._h_dim:
            raise ValueError(f"current points must have length h_dim={self._h_dim}.")
        self._current_point = current_point
        self._kernel = self.update_kernel()

    def drive_function(self, homeostatic_point: ArrayLike) -> np.ndarray:
        """現在の状態とセットポイントの差の絶対値から動因を算出する

        current_point: np.ndarray | list | float: 各次元のホメオスタシス状態

        returns: np.ndarray: 各次元ごとの動因

        """
        from scipy.special import gamma, gammainc

        homeostatic_point = np.asarray(homeostatic_point, dtype=float).reshape(-1)
        u = np.abs(self._setpoint - homeostatic_point)
        z = (u / self._rho) ** self._kappa
        # 下側不完全ガンマ γ(a,z) = Γ(a) * gammainc(a,z)
        return u - (self._rho / self._kappa) * gamma(1.0 / self._kappa) * gammainc(1.0 / self._kappa, z)

    def reward_function(
            self,
            reward_signal: ArrayLike
    ) -> np.ndarray:
        """外的報酬によるホメオスタシス状態の変化で報酬を算出する

        returns: np.ndarray: 各次元ごとの状態変化から算出された報酬

        """
        reward_signal = np.asarray(reward_signal, dtype=float).reshape(-1)
        if reward_signal.size != self._h_dim:
            raise ValueError(
                f"reward_signal must have length h_dim={self._h_dim}. "
                f"Got reward_signal={reward_signal.size}."
            )
        self._previous_point = self._current_point
        next_point = self._previous_point * self._tau + reward_signal
        self.set_current_points(next_point)
        reward = self.drive_function(self._previous_point) - self.drive_function(self._current_point)
        return reward

    def induction_function(self, reward_signal: ArrayLike):
        """外的報酬によって誘導される内的反応

        今はreward_signalをそのまま返す（rewardによって一定強度の異なる反応が生じる）
        将来的に誘導される反応がreward間で共通する成分と異なる成分、そして内的状態に依存して強度の調整が入るかもしれない

        """
        reward_signal = np.asarray(reward_signal, dtype=float).reshape(-1)
        return reward_signal

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

    def update_weight_o(self, delta: ArrayLike, action: ArrayLike):
        delta = np.asarray(delta, dtype=float).reshape(-1)
        action = np.asarray(action, dtype=float).reshape(-1)

        if action.size != self._o_dim:
            raise ValueError(f"action must have length o_dim={self._o_dim}.")

        if delta.size != self._o_dim:
            raise ValueError(f"delta must have length o_dim={self._o_dim}.")

        delta_o = delta * action  # (o_dim,)

        self._o_weights += np.einsum(
            "h,hk,o->hko",
            self._alpha,
            self.kernel,
            delta_o
        )

    def update_weight_c(self, delta: ArrayLike):
        """潜在的反応の重み更新"""
        delta = np.asarray(delta, dtype=float).reshape(-1)

        if delta.size != self._c_dim:
            raise ValueError(
                f"delta must have length c_dim={self._c_dim}. "
                f"Got delta={delta.size}."
            )

        self._c_weights += np.einsum(
            "h,hk,c->hkc",
            self._alpha,
            self.kernel,
            delta
        )
