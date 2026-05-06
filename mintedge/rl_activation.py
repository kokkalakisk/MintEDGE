from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ServerActivationAction:
    """One topology-independent action in the server activation MDP."""

    kind: str
    server: Optional[str] = None


@dataclass(frozen=True)
class ServerActivationDecision:
    """Decision returned by the activation policy."""

    status_vector: Dict[str, int]
    action: ServerActivationAction
    q_value: float
    exploratory: bool


@dataclass
class RLActivationConfig:
    """Configuration for the lightweight activation learner.

    The learner deliberately uses normalized global and per-server features,
    not base-station IDs, so learned weights can be reused on different
    topologies and service mixes.
    """

    learning_rate: float = 0.01
    discount_factor: float = 0.95
    epsilon: float = 0.1
    min_epsilon: float = 0.01
    epsilon_decay: float = 0.999
    training: bool = False
    reward_energy_weight: float = 0.05
    reward_rejection_weight: float = 10.0
    reward_qos_weight: float = 10.0
    reward_delay_weight: float = 2.0
    reward_switch_weight: float = 0.1
    min_active_servers: int = 1
    boot_time_reference: float = 300.0
    model_path: Optional[str] = None
    autosave_decisions: int = 25
    seed: Optional[int] = None


class QLearningServerActivationPolicy:
    """Linear Q-learning policy for server activation/deactivation.

    This is intentionally small enough to run inside MintEDGE without adding a
    deep-learning dependency, while still being a trainable RL model rather
    than a rule-based threshold policy. It evaluates candidate actions using
    Q(s, a) = w * phi(s, a), where phi contains normalized topology, workload,
    KPI, and candidate-server features.
    """

    ACTION_HOLD = "hold"
    ACTION_TURN_ON = "turn_on"
    ACTION_TURN_OFF = "turn_off"

    FEATURE_NAMES = (
        "bias",
        "global_demand_capacity_ratio",
        "global_active_capacity_share",
        "global_active_server_share",
        "global_rejection_rate",
        "global_qos_violation_rate",
        "global_delay_pressure",
        "global_energy_pressure",
        "action_hold",
        "action_turn_on",
        "action_turn_off",
        "result_active_capacity_share",
        "result_active_server_share",
        "candidate_is_on",
        "candidate_capacity_share",
        "candidate_idle_power_share",
        "candidate_max_power_share",
        "candidate_local_workload_share",
        "candidate_reachable_workload_share",
        "candidate_mean_sigma",
        "candidate_boot_time",
        "candidate_switch_cooldown",
    )

    def __init__(
        self,
        config: Optional[RLActivationConfig] = None,
        weights: Optional[Sequence[float]] = None,
    ):
        self.config = config or RLActivationConfig()
        self.weights = (
            np.zeros(len(self.FEATURE_NAMES), dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        if len(self.weights) != len(self.FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(self.FEATURE_NAMES)} RL weights, got {len(self.weights)}"
            )

        self.rng = np.random.default_rng(self.config.seed)
        self._pending_features: Optional[np.ndarray] = None
        self._pending_reward = 0.0
        self._pending_steps = 0
        self._decisions_since_save = 0

    @classmethod
    def from_settings(cls) -> "QLearningServerActivationPolicy":
        """Build a policy from settings.py and load weights if present."""

        import settings

        config = RLActivationConfig(
            learning_rate=getattr(settings, "RL_ACTIVATION_LEARNING_RATE", 0.01),
            discount_factor=getattr(settings, "RL_ACTIVATION_DISCOUNT_FACTOR", 0.95),
            epsilon=getattr(settings, "RL_ACTIVATION_EPSILON", 0.1),
            min_epsilon=getattr(settings, "RL_ACTIVATION_MIN_EPSILON", 0.01),
            epsilon_decay=getattr(settings, "RL_ACTIVATION_EPSILON_DECAY", 0.999),
            training=getattr(settings, "RL_ACTIVATION_TRAINING", False),
            reward_energy_weight=getattr(
                settings, "RL_ACTIVATION_REWARD_ENERGY_WEIGHT", 0.05
            ),
            reward_rejection_weight=getattr(
                settings, "RL_ACTIVATION_REWARD_REJECTION_WEIGHT", 10.0
            ),
            reward_qos_weight=getattr(
                settings, "RL_ACTIVATION_REWARD_QOS_WEIGHT", 10.0
            ),
            reward_delay_weight=getattr(
                settings, "RL_ACTIVATION_REWARD_DELAY_WEIGHT", 2.0
            ),
            reward_switch_weight=getattr(
                settings, "RL_ACTIVATION_REWARD_SWITCH_WEIGHT", 0.1
            ),
            min_active_servers=getattr(settings, "RL_ACTIVATION_MIN_ACTIVE_SERVERS", 1),
            boot_time_reference=getattr(
                settings, "RL_ACTIVATION_BOOT_TIME_REFERENCE", 300.0
            ),
            model_path=getattr(
                settings, "RL_ACTIVATION_MODEL_PATH", "./rl_activation_model.json"
            ),
            autosave_decisions=getattr(
                settings, "RL_ACTIVATION_AUTOSAVE_DECISIONS", 25
            ),
            seed=getattr(settings, "RL_ACTIVATION_SEED", None),
        )

        policy = cls(config)
        if config.model_path and Path(config.model_path).exists():
            policy.load(config.model_path)
        return policy

    def select_status(
        self,
        infr,
        demand_mat: Dict[str, Dict[str, int]],
        latest_kpis=None,
        current_status: Optional[Dict[str, int]] = None,
    ) -> ServerActivationDecision:
        """Choose a server status vector for the current state."""

        status = self._normalise_status(infr, current_status)
        self._learn_from_pending_transition(infr, demand_mat, latest_kpis, status)

        actions = self._candidate_actions(infr, status)
        features = [
            self._feature_vector(infr, demand_mat, latest_kpis, status, action)
            for action in actions
        ]
        q_values = np.asarray([float(feature @ self.weights) for feature in features])

        exploratory = bool(
            self.config.training
            and len(actions) > 1
            and self.rng.random() < self.config.epsilon
        )
        if exploratory:
            selected = int(self.rng.integers(0, len(actions)))
        else:
            selected = int(np.argmax(q_values))

        action = actions[selected]
        next_status = self._apply_action(infr, status, action)

        if self.config.training:
            self._pending_features = features[selected].copy()
            self._pending_reward = -self.config.reward_switch_weight * self._switches(
                status, next_status
            )
            self._pending_steps = 0
            self.config.epsilon = max(
                self.config.min_epsilon,
                self.config.epsilon * self.config.epsilon_decay,
            )
            self._decisions_since_save += 1
            self._autosave_if_needed()

        return ServerActivationDecision(
            status_vector=next_status,
            action=action,
            q_value=float(q_values[selected]),
            exploratory=exploratory,
        )

    def observe_kpis(self, latest_kpis, infr) -> None:
        """Accumulate reward from measured KPIs after the latest action."""

        if not self.config.training or self._pending_features is None:
            return
        self._pending_reward += self._reward_from_kpis(latest_kpis, infr)
        self._pending_steps += 1

    def save(self, path: Optional[str] = None) -> None:
        """Persist model weights and feature metadata as JSON."""

        target_path = path or self.config.model_path
        if not target_path:
            return
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": list(self.FEATURE_NAMES),
            "weights": self.weights.tolist(),
            "config": asdict(self.config),
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        """Load model weights from JSON."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        feature_names = payload.get("feature_names")
        if feature_names != list(self.FEATURE_NAMES):
            raise ValueError(
                "RL activation model feature set does not match this MintEDGE version"
            )
        weights = np.asarray(payload["weights"], dtype=float)
        if len(weights) != len(self.FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(self.FEATURE_NAMES)} RL weights, got {len(weights)}"
            )
        self.weights = weights

    def _learn_from_pending_transition(
        self,
        infr,
        demand_mat: Dict[str, Dict[str, int]],
        latest_kpis,
        status: Dict[str, int],
    ) -> None:
        if (
            not self.config.training
            or self._pending_features is None
            or self._pending_steps == 0
        ):
            return

        actions = self._candidate_actions(infr, status)
        next_q = max(
            float(
                self._feature_vector(infr, demand_mat, latest_kpis, status, action)
                @ self.weights
            )
            for action in actions
        )
        reward = self._pending_reward / self._pending_steps
        current_q = float(self._pending_features @ self.weights)
        target = reward + self.config.discount_factor * next_q
        td_error = target - current_q
        self.weights += self.config.learning_rate * td_error * self._pending_features
        self._pending_reward = 0.0
        self._pending_steps = 0

    def _candidate_actions(
        self, infr, status: Dict[str, int]
    ) -> Sequence[ServerActivationAction]:
        actions = [ServerActivationAction(self.ACTION_HOLD)]
        server_names = self._server_names(infr)
        active_servers = sum(status[name] for name in server_names)
        min_active = min(self.config.min_active_servers, len(server_names))

        for name in server_names:
            if status[name] == 1 and active_servers > min_active:
                actions.append(ServerActivationAction(self.ACTION_TURN_OFF, name))
            elif status[name] == 0:
                actions.append(ServerActivationAction(self.ACTION_TURN_ON, name))
        return actions

    def _feature_vector(
        self,
        infr,
        demand_mat: Dict[str, Dict[str, int]],
        latest_kpis,
        status: Dict[str, int],
        action: ServerActivationAction,
    ) -> np.ndarray:
        server_names = self._server_names(infr)
        server_count = max(len(server_names), 1)
        total_capacity = self._total_capacity(infr)
        total_max_power = self._total_max_power(infr)
        total_workload = self._total_workload(infr, demand_mat)
        active_capacity = self._active_capacity(infr, status)
        active_servers = sum(status[name] for name in server_names)
        result_status = self._apply_action(infr, status, action)
        result_capacity = self._active_capacity(infr, result_status)
        result_servers = sum(result_status[name] for name in server_names)

        candidate = self._candidate_features(infr, demand_mat, action, total_workload)

        features = [
            1.0,
            self._bounded(total_workload / total_capacity),
            self._bounded(active_capacity / total_capacity),
            self._bounded(active_servers / server_count),
            self._bounded(self._rejection_rate(latest_kpis)),
            self._bounded(self._qos_violation_rate(latest_kpis)),
            self._bounded(self._observed_delay_pressure(infr, latest_kpis)),
            self._bounded(self._energy_pressure(latest_kpis, total_max_power)),
            1.0 if action.kind == self.ACTION_HOLD else 0.0,
            1.0 if action.kind == self.ACTION_TURN_ON else 0.0,
            1.0 if action.kind == self.ACTION_TURN_OFF else 0.0,
            self._bounded(result_capacity / total_capacity),
            self._bounded(result_servers / server_count),
        ]
        features.extend(candidate)
        return np.asarray(features, dtype=float)

    def _candidate_features(
        self,
        infr,
        demand_mat: Dict[str, Dict[str, int]],
        action: ServerActivationAction,
        total_workload: float,
    ) -> Sequence[float]:
        if action.server is None:
            return [0.0] * 9

        bs = infr.bss[action.server]
        server = bs.server
        total_capacity = self._total_capacity(infr)
        total_idle_power = self._total_idle_power(infr)
        total_max_power = self._total_max_power(infr)
        local_workload = self._bs_workload(infr, demand_mat, action.server)
        reachable_workload = self._reachable_workload(infr, demand_mat, action.server)
        mean_sigma = self._weighted_mean_sigma(infr, demand_mat, action.server)

        boot_time = float(server.boot_time or 0)
        if boot_time > 0:
            switch_cooldown = min(1.0, (infr.env.now - server.last_onoff_time) / boot_time)
        else:
            switch_cooldown = 1.0

        return [
            1.0 if server.is_on else 0.0,
            self._bounded(server.max_cap / total_capacity),
            self._bounded(server.idle_power / total_idle_power),
            self._bounded(server.max_power / total_max_power),
            self._bounded(local_workload / max(total_workload, 1.0)),
            self._bounded(reachable_workload / max(total_workload, 1.0)),
            self._bounded(mean_sigma),
            self._bounded(boot_time / max(self.config.boot_time_reference, 1.0)),
            self._bounded(switch_cooldown),
        ]

    def _reward_from_kpis(self, latest_kpis, infr) -> float:
        energy_kw = (
            self._kpi_value(latest_kpis, "dynamic_W_servers")
            + self._kpi_value(latest_kpis, "idle_W_servers")
            + self._kpi_value(latest_kpis, "W_links")
        ) / 1000.0
        rejection_rate = self._rejection_rate(latest_kpis)
        qos_rate = self._qos_violation_rate(latest_kpis)
        delay_pressure = self._observed_delay_pressure(infr, latest_kpis)

        return -(
            self.config.reward_energy_weight * energy_kw
            + self.config.reward_rejection_weight * rejection_rate
            + self.config.reward_qos_weight * qos_rate
            + self.config.reward_delay_weight * delay_pressure
        )

    def _normalise_status(
        self, infr, status: Optional[Dict[str, int]]
    ) -> Dict[str, int]:
        normalised = {}
        for bs in infr.bss.values():
            if bs.server is None:
                normalised[bs.name] = 0
            elif status and bs.name in status:
                normalised[bs.name] = 1 if status[bs.name] else 0
            else:
                normalised[bs.name] = 1 if bs.server.is_on else 0
        return normalised

    def _apply_action(
        self, infr, status: Dict[str, int], action: ServerActivationAction
    ) -> Dict[str, int]:
        next_status = self._normalise_status(infr, status)
        if action.server is not None and action.server in next_status:
            if action.kind == self.ACTION_TURN_ON:
                next_status[action.server] = 1
            elif action.kind == self.ACTION_TURN_OFF:
                next_status[action.server] = 0
        return next_status

    def _switches(self, old_status: Dict[str, int], new_status: Dict[str, int]) -> int:
        return sum(
            1
            for name in set(old_status) | set(new_status)
            if old_status.get(name, 0) != new_status.get(name, 0)
        )

    def _server_names(self, infr) -> Sequence[str]:
        return [bs.name for bs in infr.bss.values() if bs.server is not None]

    def _total_capacity(self, infr) -> float:
        return max(
            sum(bs.server.max_cap for bs in infr.bss.values() if bs.server is not None),
            1.0,
        )

    def _active_capacity(self, infr, status: Dict[str, int]) -> float:
        return sum(
            bs.server.max_cap
            for bs in infr.bss.values()
            if bs.server is not None and status.get(bs.name, 0) == 1
        )

    def _total_idle_power(self, infr) -> float:
        return max(
            sum(bs.server.idle_power for bs in infr.bss.values() if bs.server is not None),
            1.0,
        )

    def _total_max_power(self, infr) -> float:
        return max(
            sum(bs.server.max_power for bs in infr.bss.values() if bs.server is not None),
            1.0,
        )

    def _total_workload(self, infr, demand_mat: Dict[str, Dict[str, int]]) -> float:
        return sum(
            demand_mat.get(bs.name, {}).get(serv.name, 0) * serv.workload
            for bs in infr.bss.values()
            for serv in infr.services.values()
        )

    def _bs_workload(
        self, infr, demand_mat: Dict[str, Dict[str, int]], bs_name: str
    ) -> float:
        return sum(
            demand_mat.get(bs_name, {}).get(serv.name, 0) * serv.workload
            for serv in infr.services.values()
        )

    def _reachable_workload(
        self, infr, demand_mat: Dict[str, Dict[str, int]], dst_name: str
    ) -> float:
        dst = infr.bss[dst_name]
        if dst.server is None:
            return 0.0

        reachable = 0.0
        for src in infr.bss.values():
            for serv in infr.services.values():
                req = demand_mat.get(src.name, {}).get(serv.name, 0)
                if req <= 0:
                    continue
                if self._can_reach_within_delay(infr, src, dst, serv):
                    reachable += req * serv.workload
        return reachable

    def _can_reach_within_delay(self, infr, src, dst, serv) -> bool:
        if dst.server is None:
            return False
        try:
            t_r = infr.get_path_delay(src, dst, serv)
            t_o = infr.get_path_out_delay(dst, src, serv)
        except KeyError:
            return False
        t_u = src.get_delay(serv.input_size)
        t_d = src.get_delay(serv.output_size)
        t_c = serv.workload / max(dst.server.max_cap, 1)
        return t_u + t_r + t_o + t_d + t_c <= serv.max_delay

    def _weighted_mean_sigma(
        self, infr, demand_mat: Dict[str, Dict[str, int]], dst_name: str
    ) -> float:
        max_sigma = max(
            (infr.get_path_sigma(path) for path in infr.paths.values()),
            default=1.0,
        )
        weighted_sigma = 0.0
        total_req = 0.0
        for src_name, service_demand in demand_mat.items():
            req = sum(service_demand.values())
            if req <= 0:
                continue
            path = infr.paths.get((src_name, dst_name))
            if path is None:
                continue
            weighted_sigma += req * infr.get_path_sigma(path)
            total_req += req
        if total_req == 0:
            return 0.0
        return weighted_sigma / total_req / max(max_sigma, 1e-12)

    def _rejection_rate(self, latest_kpis) -> float:
        total_requests = self._kpi_value(latest_kpis, "total_requests")
        if total_requests <= 0:
            return 0.0
        return self._kpi_value(latest_kpis, "total_rejected") / total_requests

    def _qos_violation_rate(self, latest_kpis) -> float:
        total_requests = self._kpi_value(latest_kpis, "total_requests")
        if total_requests <= 0:
            return 0.0
        return self._sum_kpis(latest_kpis, "unsatisf_req_") / total_requests

    def _observed_delay_pressure(self, infr, latest_kpis) -> float:
        pressure = 0.0
        for bs in infr.bss:
            for serv in infr.services.values():
                delay = self._kpi_value(latest_kpis, f"delay_{bs}_{serv.name}")
                if delay > 0:
                    pressure = max(pressure, max(0.0, delay / serv.max_delay - 1.0))
        return pressure

    def _energy_pressure(self, latest_kpis, total_max_power: float) -> float:
        energy = (
            self._kpi_value(latest_kpis, "dynamic_W_servers")
            + self._kpi_value(latest_kpis, "idle_W_servers")
            + self._kpi_value(latest_kpis, "W_links")
        )
        return energy / max(total_max_power, 1.0)

    def _kpi_value(self, latest_kpis, column: str) -> float:
        if latest_kpis is None:
            return 0.0
        try:
            if hasattr(latest_kpis, "empty"):
                if latest_kpis.empty or column not in latest_kpis.columns:
                    return 0.0
                value = latest_kpis[column].iloc[-1]
            elif isinstance(latest_kpis, dict):
                value = latest_kpis.get(column, 0.0)
            else:
                return 0.0
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return 0.0
            return float(value)
        except (KeyError, TypeError, ValueError):
            return 0.0

    def _sum_kpis(self, latest_kpis, prefix: str) -> float:
        if latest_kpis is None:
            return 0.0
        columns: Iterable[str]
        if hasattr(latest_kpis, "columns"):
            columns = latest_kpis.columns
        elif isinstance(latest_kpis, dict):
            columns = latest_kpis.keys()
        else:
            return 0.0
        return sum(self._kpi_value(latest_kpis, column) for column in columns if column.startswith(prefix))

    def _bounded(self, value: float, upper: float = 5.0) -> float:
        if value is None or math.isnan(value) or math.isinf(value):
            return 0.0
        return max(0.0, min(float(value), upper))

    def _autosave_if_needed(self) -> None:
        if (
            not self.config.model_path
            or self.config.autosave_decisions <= 0
            or self._decisions_since_save < self.config.autosave_decisions
        ):
            return
        self.save()
        self._decisions_since_save = 0
