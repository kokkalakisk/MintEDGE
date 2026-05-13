import sys
import itertools
import logging
import math
import copy
import settings
from tqdm import tqdm
from collections import defaultdict
from .infrastructure import (
    Infrastructure,
    BaseStation,
    Link,
)
from .services import Service
from typing import Dict, List, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
formatter = logging.Formatter("| %(levelname)s | %(message)s")
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.WARNING)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


class EnergyHeuristic:
    __slots__ = ["infr"]

    def __init__(self, infr: Infrastructure):
        """Heuristic responsible for the energy optimization of the
        infrastructure
        Args:
            infr (Infrastructure): Infrastructure to optimize
        """
        self.infr = infr

    def get_allocation(
        self,
        demand_matrix: Dict[str, Dict[str, int]],
        server_status: Optional[Dict[str, int]] = None,
    ):
        """Compute server status, request routing, and CPU allocation.

        This is the orchestrator-facing entrypoint. When an external status
        vector is provided, for example by the RL activation policy, the
        heuristic respects that status and only computes routing/allocation
        over the active servers.
        """
        if server_status is not None:
            return self.always_on(demand_matrix, server_status)

        strategy = getattr(settings, "ENERGY_HEURISTIC_STRATEGY", "odesa").lower()
        if strategy == "always_on":
            return self.always_on(demand_matrix)
        if strategy == "threshold_based":
            threshold = getattr(settings, "ENERGY_HEURISTIC_THRESHOLD", 0.2)
            return self.threshold_based(demand_matrix, threshold)
        if strategy != "odesa":
            raise ValueError(f"Unknown ENERGY_HEURISTIC_STRATEGY: {strategy}")
        return self.odesa(demand_matrix)

    def _calculate_energy_on(
        self,
        bsj: BaseStation,
        gamma: Dict[str, Dict[str, Dict[str, float]]],
        lmbda: Dict[str, Dict[str, float]],
    ):
        return bsj.server.max_power + sum(
            (
                gamma[bs][a.name][bsj.name]
                * bsj.server.op_energy
                * a.workload
                * lmbda[bs][a.name]
            )
            + (
                gamma[bs][a.name][bsj.name]
                * self.infr.get_path_sigma(self.infr.paths[(bs, bsj.name)])
                * a.input_size
                * lmbda[bs][a.name]
            )
            for bs in self.infr.bss
            for a in self.infr.services.values()
        )

    def always_on(
        self,
        lmbda: Dict[str, Dict[str, float]],
        server_status: Optional[Dict[str, int]] = None,
    ):
        """Heuristic to allocate resources on the infrastructure
        Args:
            lmbda (Dict[str, Dict[str, float]]): Demand matrix
        Returns:
            Dict[str, int]: on/off state of the servers
            Dict[str, Dict[str, Dict[str, float]]]: Request routing
            Dict[str, Dict[str, Dict[str, float]]]: Resource allocation
        """
        infr = self.infr

        eta, gamma, beta, _ = self.initialize(lmbda, infr, server_status)

        return eta, gamma, beta

    def odesa(self, lmbda: Dict[str, Dict[str, float]]):
        """Heuristic to allocate resources on the infrastructure
        Args:
            lmbda (Dict[str, Dict[str, float]]): Demand matrix
        Returns:
            Dict[str, int]: on/off state of the servers
            Dict[str, Dict[str, Dict[str, float]]]: Request routing
            Dict[str, Dict[str, Dict[str, float]]]: Resource allocation
        """
        infr = self.infr

        eta, gamma, beta, used_cap = self.initialize(lmbda, infr)
        # Short edge servers according to their idle energy consumption
        sorted_servers = self._sort_servers_by_e_idle(infr.bss)
        # Short services according to their delay budgets (ascending)
        sorted_services = self.sort_services()

        # Main loop
        for bsj in tqdm(sorted_servers, leave=False, desc="Energy Heuristic"):
            # Keep copies of the current status just in case I can't turn it off
            gamma_bckp = copy.deepcopy(gamma)
            used_cap_bckp = copy.deepcopy(used_cap)
            # Get current energy consumption
            e_on = self._calculate_energy_on(bsj, gamma, lmbda)
            e_off = 0
            eta[bsj.name] = 0  # Turn off the server

            # Reroute requests to other servers
            for a, bsi in itertools.product(sorted_services, infr.bss.values()):
                # if no requests from bsi to a attended by bsj, skip
                if gamma[bsi.name][a.name][bsj.name] == 0:
                    continue
                # Get servers where I can route
                cand = self._get_cand_servers(eta, bsi, a)
                # bsj is not a candidate (i'm turning it off)
                if bsj in cand:
                    cand.remove(bsj)
                # If no servers to route to, break and try shutting down next server
                if len(cand) == 0:
                    eta[bsj.name] = 1
                    gamma = copy.deepcopy(gamma_bckp)
                    used_cap = copy.deepcopy(used_cap_bckp)
                    break
                # Requests to this service coming from this bs that need relocation
                req_to_reroute = math.ceil(
                    lmbda[bsi.name][a.name] * gamma[bsi.name][a.name][bsj.name]
                )
                # Reroute requests
                gamma, req_to_reroute, used_cap, e_route = self._reroute(
                    cand, bsi, a, req_to_reroute, lmbda, gamma, used_cap
                )

                # The off should not have requests routed to it
                gamma[bsi.name][a.name][bsj.name] = 0
                used_cap[bsj.name] = 0
                # Update energy consumption of turning the server off
                e_off += e_route
                # Have all requests been routed?
                # or is the routing energy more than keeping the server on?
                if req_to_reroute > 0 or e_off >= e_on:
                    eta[bsj.name] = 1  # if not, turn the server back on
                    # restore the previous routing
                    gamma = copy.deepcopy(gamma_bckp)
                    used_cap = copy.deepcopy(used_cap_bckp)
                    # Try turning off next server
                    break

        # Calcualte resource allocation
        beta = self._calculate_beta_matrix(lmbda, gamma, infr)
        # Check delay constraints
        delays_ok, viol_src, a, viol_dst = self._check_delays(gamma, beta)
        if not delays_ok:
            # Turn on the server with the delay constraint violated
            eta, gamma, beta = self._handle_violated_delays(
                viol_src, a, viol_dst, lmbda, eta, gamma, used_cap
            )

        # self._check_results(lmbda, gamma, beta, infr)
        return eta, gamma, beta

    def _handle_violated_delays(
        self, viol_src, a, viol_dst, lmbda, eta, gamma, used_cap
    ):
        """Handle the case where the delay constraints are violated
        Args:
            viol (BaseStation):
            lmbda (Dict[str, Dict[str, int]]): Request matrix
            eta (Dict[str, int]): on/off matrix
            gamma (Dict[str, Dict[str, Dict[str, float]]]): Current routing matrix
            used_cap (Dict[str, float]): capacity matrix
        Returns:
            New beta and gamma matrixes
        """
        delays_ok = False
        count = 0
        while not delays_ok:
            if count > settings.MAX_DELAY_CHECKS:
                break
            count += 1
            # Get candidates to unload the server with the delay constraint violated
            # sort them by free capacity
            cands = self._sort_servers_by_free_capacity(
                used_cap, self._get_cand_servers(eta, viol_src, a)
            )
            # Keep copies of the current status just in case I can't turn it off
            gamma_bckp = copy.deepcopy(gamma)
            used_cap_bckp = copy.deepcopy(used_cap)
            # Try to unload to the emptiest server
            try:
                new_dst = cands[0] if cands[0] != viol_dst else cands[1]
            except IndexError:
                from mintedge import RAND_NUM_GEN

                new_dst = RAND_NUM_GEN.choice(
                    [
                        bs
                        for bs in self.infr.bss.values()
                        if bs != viol_src and bs.server is not None
                    ]
                )
                eta[new_dst.name] = 1

            avail_cap = new_dst.server.max_cap - used_cap[new_dst.name]
            req_to_route = math.ceil(
                lmbda[viol_src.name][a.name]
                * gamma[viol_src.name][a.name][viol_dst.name]
            )
            # Is there enouth capacity in that server to unload all requests?
            if req_to_route <= self._ops_to_req(avail_cap, a):
                # Yes, unload all requests
                # previous routing to the new server
                gamma[viol_src.name][a.name][new_dst.name] += gamma[
                    viol_src.name
                ][a.name][viol_dst.name]
                # unload the old server
                gamma[viol_src.name][a.name][viol_dst.name] = 0
                # update used capacity
                used_cap[new_dst.name] += self._req_to_ops(req_to_route, a)
                # calculate new beta
                beta = self._calculate_beta_matrix(lmbda, gamma, self.infr)
                # calculate new delay
                new_delay = self._calculate_delay(viol_src, a, new_dst, beta)
                # Is the delay constraint still violated?
                if new_delay > a.max_delay:
                    # Yes, restore previous routing and turn on a new server
                    gamma = copy.deepcopy(gamma_bckp)
                    used_cap = copy.deepcopy(used_cap_bckp)

                    # Turn the server of the BS with violations on (if it is not)
                    if (
                        viol_src.server is not None
                        and eta[viol_src.server.name] == 0
                    ):
                        eta[viol_src.name] = 1
                        new_dst = viol_src

                    else:  # if no server or server is already on, turn on se
                        servs = self.infr.bss.values()
                        for bs in servs:
                            if eta[bs.name] == 0 and bs.server is not None:
                                eta[bs.name] = 1
                                new_dst = bs
                                break
            else:
                # No space, turn on a server
                # Turn the server of the BS with violations on (if it is not)
                if (
                    viol_src.server is not None
                    and eta[viol_src.server.name] == 0
                ):
                    eta[viol_src.name] = 1

                else:  # if no server or server is already on, turn on se
                    servs = self.sort_servers_by_sigma(
                        self.infr.bss.values(), viol_src, self.infr
                    )
                    for bs in servs:
                        if eta[bs.name] == 0 and bs.server is not None:
                            eta[bs.name] = 1
                            break
                continue  # go to the next iteration to evaluate if this new server is enough

            # Check if the new routing is okay to fulfil the delay constraint
            beta = self._calculate_beta_matrix(lmbda, gamma, self.infr)
            delays_ok, viol_src, a, viol_dst = self._check_delays(gamma, beta)
        beta = self._calculate_beta_matrix(lmbda, gamma, self.infr)
        return eta, gamma, beta

    def _sort_servers_by_free_capacity(
        self, used_cap: Dict[str, float], servers: List[BaseStation]
    ):
        """Sorts the servers by their free capacity
        Args:
            used_cap (Dict[str, float]): Used capacity in each server
            servers (List[BaseStation]): List of servers to sort
        Returns:
            List of servers sorted by free capacity
        """
        return sorted(
            servers,
            key=lambda bs: used_cap[bs.name] / bs.server.max_cap,
            reverse=False,
        )

    def _reroute(
        self,
        candidates: List[BaseStation],
        src: BaseStation,
        a: Service,
        req_to_route: int,
        lmbda: Dict[str, Dict[str, int]],
        gamma: Dict[str, Dict[str, Dict[str, float]]],
        used_cap: Dict[str, float],
    ):
        """Routes requests from src to a to the servers in candidates depending on
        their capacity and that of the links.
        Args:
            candidates (List[BaseStation]): List of servers to route to
            src (BaseStation): Source server
            a (Service): Service being routed
            req_to_route (int): Number of requests to route
            lmbda (Dict[str, Dict[str, int]]): Request matrix
            gamma (Dict[str, Dict[str, Dict[str, float]]]): Current routing matrix
            used_cap (Dict[str, float]): Used capacity in each server
        Returns:
            gamma (Dict[str, Dict[str, Dict[str, float]]]): Updated routing matrix
            req_to_route (int): Number of requests not routed
            used_cap (Dict[str, float]): Updated used capacity in each server
            e_route (float): Energy consumed in the routing
        """
        e_route = 0
        for dst in candidates:
            # if there are no more requests to route, stop
            if req_to_route == 0:
                break
            # Get the path to destination
            path = self.infr.paths[(src.name, dst.name)]
            # Is there enough capacity in the path to dst?
            alpha = self._calculate_alpha(lmbda, gamma, path)
            data = self._req_to_bits(req_to_route, a)
            assig_req = 0  # Number of requests assigned to dst (initialy 0)
            if data <= alpha:
                # Is there enough capacity in the destination server?
                avail_cap = dst.server.max_cap - used_cap[dst.name]
                if req_to_route <= self._ops_to_req(avail_cap, a):
                    # Enough capacity: Assign all to dst
                    assig_req = req_to_route
                    req_to_route = 0
                else:
                    # Not enough capacity: Assign as much as possible
                    assig_req = self._ops_to_req(avail_cap, a)
                    req_to_route -= assig_req
            else:
                # assign as much as possible
                assig_req = self._bits_to_req(alpha, a)
                # Is there enough capacity for assig_req in the server?
                avail_cap = dst.server.max_cap - used_cap[dst.name]
                if assig_req >= self._ops_to_req(avail_cap, a):
                    # Not enough capacity: Assign as much as possible
                    assig_req = self._ops_to_req(avail_cap, a)
                # if there was enough capacity I don't need to do anything.
                # assig_req is as much as I can send through the link
                req_to_route -= assig_req
            # Update gamma according to assig_req. += because there might already be requests from srt to dst
            gamma[src.name][a.name][dst.name] += (
                assig_req / lmbda[src.name][a.name]
            )
            # Update used capacity
            used_cap[dst.name] += self._req_to_ops(assig_req, a)
            # Update energy used to route
            e_route += (
                self._req_to_ops(assig_req, a) * dst.server.op_energy
            ) + (
                self._req_to_bits(assig_req, a) * self.infr.get_path_sigma(path)
            )

        return gamma, req_to_route, used_cap, e_route

    def initialize(
        self,
        lmbda: Dict[str, Dict[str, float]],
        infr: Infrastructure,
        server_status: Optional[Dict[str, int]] = None,
    ) -> Tuple[
        Dict[str, int],
        Dict[str, Dict[str, float]],
        Dict[str, Dict[str, Dict[str, float]]],
    ]:
        """Initialize the values of the matrices for the heuristic
        Args:
            lmbda (Dict[str, Dict[str, float]]): arrival rates of services at
                each BS
            infr (Infrastructure): The infrastructure
        Return:
            eta: active edge servers
            beta: resource allocation at each edge server
            gamma: requests routing
        """
        # Initialize gamma, beta and eta
        gamma = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        beta = defaultdict(lambda: defaultdict(float))

        if server_status is None:
            eta = {
                bs.name: 1 if bs.server is not None else 0
                for bs in infr.bss.values()
            }  # all servers are active
        else:
            eta = {
                bs.name: 1
                if bs.server is not None and server_status.get(bs.name, 0) == 1
                else 0
                for bs in infr.bss.values()
            }

        used_cap = {bs: 0 for bs in infr.bss}

        # Check if there is enough capacity in the infrastructure
        total_capacity = sum(
            bs.server.max_cap
            for bs in infr.bss.values()
            if bs.server is not None and eta[bs.name] == 1
        )
        total_demand = sum(
            ak.workload * lmbda[bs][ak.name]
            for bs in infr.bss
            for ak in infr.services.values()
        )

        if total_demand > total_capacity:
            raise Exception("Not enough capacity")

        # Main loop
        for bsi, a in tqdm(
            itertools.product(infr.bss.values(), infr.services.values()),
            leave=False,
            desc="Initializing resource allocation",
        ):
            # If there are no requests, skip
            if lmbda[bsi.name][a.name] == 0:
                continue

            # Requests to route this iteration
            req_to_locate = lmbda[bsi.name][a.name]

            # Get servers that can attend the requests within the constraints
            cand = self._get_cand_servers(eta, bsi, a)
            # Reroute requests
            gamma, req_to_locate, used_cap, _ = self._reroute(
                cand, bsi, a, req_to_locate, lmbda, gamma, used_cap
            )
            if req_to_locate > 0:
                raise Exception(
                    f"Could not route {req_to_locate} requests for {bsi.name},{a.name}."
                )

        beta = self._calculate_beta_matrix(lmbda, gamma, infr)

        # self._check_results(lmbda, gamma, beta, infr)

        return eta, gamma, beta, used_cap

    def _check_delays(
        self,
        gamma: Dict[str, Dict[str, Dict[str, float]]],
        beta: Dict[str, Dict[str, float]],
    ) -> bool:
        """Check if the delay constraints are met.
        Args:
            gamma (Dict[str, Dict[str, Dict[str, float]]]): Routing matrix.
            beta (Dict[str, Dict[str, Dict[str, float]]]): Resource allocation
                matrix.
            dst (BaseStation): Destination BS.
        Returns:
            bool: True if the constraints are met, False otherwise.
        """
        for src, a, dst in itertools.product(
            self.infr.bss.values(),
            self.infr.services.values(),
            self.infr.bss.values(),
        ):
            if gamma[src.name][a.name][dst.name] == 0:
                continue
            t_u = src.get_delay(a.input_size)  # RAN delay
            t_d = src.get_delay(a.output_size)  # RAN delay output
            t_r = self.infr.get_path_delay(src, dst, a)  # Backhaul
            try:
                t_c = a.workload / (beta[a.name][dst.name] * dst.server.max_cap)
                # t_c = a.workload / dst.server.max_cap

            except ZeroDivisionError:
                t_c = math.inf
            t_o = self.infr.get_path_out_delay(dst, src, a)  # Backhaul output

            if t_u + t_r + t_c + t_d + t_o > a.max_delay:
                return False, src, a, dst

        return True, None, None, None

    def _calculate_delay(
        self,
        src: BaseStation,
        a: Service,
        dst: BaseStation,
        beta: Dict[str, Dict[str, Dict[str, float]]],
    ) -> float:
        t_u = src.get_delay(a.input_size)  # RAN delay
        t_d = src.get_delay(a.output_size)  # RAN delay output
        t_r = self.infr.get_path_delay(src, dst, a)  # Backhaul
        try:
            t_c = a.workload / (beta[a.name][dst.name] * dst.server.max_cap)
        except ZeroDivisionError:
            t_c = math.inf
        t_o = self.infr.get_path_out_delay(dst, src, a)  # Backhaul output
        return t_u + t_r + t_c + t_d + t_o

    def _calculate_transport_delay(
        self,
        src: BaseStation,
        a: Service,
        dst: BaseStation,
    ) -> float:
        t_u = src.get_delay(a.input_size)  # RAN delay
        t_d = src.get_delay(a.output_size)  # RAN delay output
        t_r = self.infr.get_path_delay(src, dst, a)  # Backhaul
        t_o = self.infr.get_path_out_delay(dst, src, a)  # Backhaul output
        return t_u + t_r + t_d + t_o

    def _ops_to_req(self, ops: int, service: Service) -> int:
        """Converts operations into number of requests
        Args:
            ops (int): the number of operations to convert
            service (Service): the service of the requests evaluated
        Return:
            int: the number of requests
        """
        return math.floor(ops / service.workload)

    def _req_to_ops(self, req: int, service: Service) -> int:
        """Converts requests into number of operations
        Args:
            req (int): the number of requests to convert
            service (Service): the service of the requests evaluated
        Return:
            int: the number of operations
        """
        return req * service.workload

    def _req_to_bits(self, req: int, service: Service) -> int:
        """Converts requests into number of virtual machines
        Args:
            req (int): the number of requests to convert
            service (Service): the service of the requests evaluated
        Return:
            int: the number of bits to be transmitted
        """
        return req * service.input_size

    def _bits_to_req(self, bits: int, service: Service) -> int:
        """Converts bits into number of requests
        Args:
            bits (int): the number of bits to convert
            service (Service): the service of the requests evaluated
        Return:
            int: the number of requests
        """
        return math.floor(bits / service.input_size)

    def sort_services(self) -> List[Service]:
        """Sorts the services according to their time budget
        Returns:
            List[Service]: Sorted list of services
        """
        return sorted(
            self.infr.services.values(),
            key=lambda x: x.max_delay,
            reverse=True,
        )

    def _calculate_link_used_capacity(
        self,
        link: Link,
        infr: Infrastructure,
        lmbda: Dict[str, Dict[str, int]],
        gamma: Dict[str, Dict[str, Dict[str, float]]],
    ) -> int:
        """Get the used capacity of a link in number of operations per second
        Args:
            link (Link): The link to get the used capacity of
            infr (Infrastructure): The infrastructure
            lmbda (Dict[str, Dict[str, int]]): The request arrival matrix
            gamma (Dict[str, Dict[str, Dict[str, float]]]): The request
                routing matrix
        Return:
            int: The used capacity of the link in bps
        """
        cap = 0
        srvcs = infr.services.values()
        for (bsi, bsj), a in itertools.product(infr.paths.keys(), srvcs):
            # if bsi != bsj:
            if link in infr.paths[(bsi, bsj)]:
                cap += gamma[bsi][a.name][bsj] * lmbda[bsi][a.name] * a.input_size

        return int(cap)

    def _calculate_used_capacity(self, bsi: BaseStation, beta: Dict[str, Dict[str, float]]):
        """Calculate the capacity used by a server according to the beta matrix
        Args:
            bsi (BaseStation): The server to calculate the used capacity of
            beta (Dict[str, Dict[str, float]]): The resource allocation matrix.
        Returns:
            int: The used capacity of the server in ops per second
        """
        if bsi.server is not None:
            return sum(
                beta[service][bsi.name] * bsi.server.max_cap
                for service in beta
            )
        else:
            return 0

    def _get_cand_servers(
        self,
        eta: Dict[str, int],
        bsi: BaseStation,
        ak: Service,
    ) -> List[BaseStation]:
        """Returns a list of candidate servers for the service ak in bsi
        Args:
            eta (Dict[str, int]): Dictionary of base stations with 1 if they
                are on
            bsi (BaseStation): Base station where the requests are received
            ak (str): Service name
        Returns:
            List[BaseStation]: List of servers that can attend the requests.
        """
        cand_servers = []

        # Get all paths from bsi to all other base stations.
        cand_paths = {
            k: v for k, v in self.infr.paths.items() if k[0] == bsi.name
        }
        # Get BSs with a server and within the delay budget
        for path in cand_paths:
            dst = self.infr.bss[path[1]]
            if dst.server is not None and eta[dst.name] == 1:
                t_u = bsi.get_delay(ak.input_size)
                t_c = self._calculate_comput_delay(ak, dst.server.max_cap)
                t_r = self.infr.get_path_delay(bsi, dst, ak)
                rem_delay_budget = ak.max_delay - t_u - t_c

                if t_r < rem_delay_budget:
                    cand_servers += [dst]
        # Order the servers according to the sigma of their paths
        cand_servers = self.sort_servers_by_sigma(cand_servers, bsi, self.infr)

        return cand_servers

    def _calculate_comput_delay(self, ak: Service, max_cap: int) -> float:
        """Calculates the computing delay of a service according to the
        number of requests attended
        Args:
            ak (Service): Service to calculate the delay of
            max_cap (int): Maximum number of operations per second
        Returns:
            float: Computing delay of the service"""
        return ak.workload / max_cap

    def _sort_servers_by_e_idle(
        self, bss: Dict[str, BaseStation]
    ) -> List[BaseStation]:
        """Sorts a list of servers according to the sigma of the path from src
        Args:
            bss (Dict[str, BaseStation]): List of BSs to sort
        Returns:
            List[BaseStation]: Sorted list of servers
        """
        return sorted(
            [v for v in bss.values() if v.server is not None],
            key=lambda x: x.server.idle_power,
            reverse=True,
        )

    def _sort_servers_by_e_op(
        self, bss: Dict[str, BaseStation]
    ) -> List[BaseStation]:
        """Sorts a list of servers according to the sigma of the path from src
        Args:
            bss (Dict[str, BaseStation]): List of BSs to sort
        Returns:
            List[BaseStation]: Sorted list of servers
        """
        return sorted(
            [v for v in bss.values() if v.server is not None],
            key=lambda x: x.server.op_energy,
            reverse=True,
        )

    def sort_servers_by_sigma(
        self, bss: List[BaseStation], src: BaseStation, infr: Infrastructure
    ) -> List[BaseStation]:
        """Sorts a list of servers according to the sigma of the path from src
        Args:
            servers (List[BaseStation]): List of servers to sort
            src (BaseStation): Source base station from which the path starts
            infr (Infrastructure): The infrastructure
        Returns:
            List[BaseStation]: Sorted list of servers (decreasing order)
        """
        return sorted(
            bss,
            key=lambda x: infr.get_path_sigma(infr.paths[(src.name, x.name)]),
        )

    def _calculate_alpha_link(
        self,
        link: Link,
        lmbda: Dict[str, Dict[str, int]],
        gamma: Dict[str, Dict[str, Dict[str, float]]],
    ) -> int:
        """Get the used capacity of a link in number of operations per second
        Args:
            link (Link): The link to get the used capacity of
            lmbda (Dict[str, Dict[str, int]]): The request arrival matrix
            gamma (Dict[str, Dict[str, Dict[str, float]]]): The resource
                allocation matrix
        Return:
            int: The used capacity of the link in bps
        """
        cap = link.capacity
        # friendly name for list of services
        srvcs = self.infr.services.values()
        for (bsi, bsj), a in itertools.product(self.infr.paths.keys(), srvcs):
            path = self.infr.paths[(bsi, bsj)]
            if link in path:
                cap -= gamma[bsi][a.name][bsj] * lmbda[bsi][a.name] * a.input_size

        if cap < 0:
            raise ValueError("Link capacity is negative")

        return int(cap)

    def _calculate_alpha(
        self,
        lmbda: Dict[str, Dict[str, int]],
        gamma: Dict[str, Dict[str, Dict[str, float]]],
        path: List[Link],
    ) -> int:
        """Calculate alpha (remaining capacity) for a given path.
        Args:
            lmbda (Dict[str, Dict[str, int]]): matrix with the number of
                requests per service and bs.
            gamma (Dict[str, Dict[str, Dict[str, float]]]): routing fractions
                of the requests.
        Returns:
            int: Remaining capacity in bps.
        """
        if len(path) == 0:
            return math.inf
        return min(
            self._calculate_alpha_link(link, lmbda, gamma) for link in path
        )

    def _calculate_beta_matrix(
        self,
        lmbda: Dict[str, Dict[str, int]],
        gamma: Dict[str, Dict[str, Dict[str, float]]],
        infr: Infrastructure,
    ) -> Dict[str, Dict[str, float]]:
        """Calculates the beta matrix that represents the computing resource
        allocation.
        Args:
            lmbda (Dict[str, Dict[str, int]]): Requests per service and bs.
            gamma (Dict[str, Dict[str, Dict[str, float]]]): Requests routing.
            infr (Infrastructure): The infrastructure.
        Returns:
            Dict[str, Dict[str, float]]: The beta matrix.
        """
        beta = defaultdict(lambda: defaultdict(float))

        for bsj in infr.bss.values():
            if bsj.server is None:
                continue
            o_j = sum(
                self._req_to_ops(
                    gamma[bsi][a.name][bsj.name] * lmbda[bsi][a.name], a
                )
                for bsi in infr.bss
                for a in infr.services.values()
            )
            if round(o_j) > bsj.server.max_cap:
                raise Exception(
                    f"{o_j} exceeds {bsj.name} capacity {bsj.server.max_cap}."
                )
            if o_j == 0:
                continue
            for a in infr.services.values():
                total_req = sum(
                    gamma[b][a.name][bsj.name] * lmbda[b][a.name]
                    for b in infr.bss
                )
                beta[a.name][bsj.name] = (
                    self._req_to_ops(total_req, a) / bsj.server.max_cap  # o_j
                )
                if beta[a.name][bsj.name] > 0:
                    max_trans_delay = max(
                        self._calculate_transport_delay(infr.bss[bsi], a, bsj)
                        for bsi in gamma.keys()
                        if gamma[bsi][a.name][bsj.name] > 0
                    )
                    t = a.max_delay - max_trans_delay

                    beta[a.name][bsj.name] = max(
                        beta[a.name][bsj.name],
                        a.workload / (t * bsj.server.max_cap),
                    )

            total_beta = sum(beta[a][bsj.name] for a in infr.services)

            if total_beta > 1:
                excess = total_beta - 1
                services = [
                    a
                    for a in infr.services.values()
                    if beta[a.name][bsj.name] > 0
                ]
                division = excess / len(services)
                length = len(services)
                for i, a in enumerate(services):
                    min_beta = a.workload / (t * bsj.server.max_cap)
                    if beta[a.name][bsj.name] - division < min_beta:
                        division += division / (length - i)
                        continue
                    beta[a.name][bsj.name] -= division
            else:
                remaining = 1 - total_beta
                services = [
                    a
                    for a in infr.services.values()
                    if beta[a.name][bsj.name] > 0
                ]
                for a in services:
                    min_beta = a.workload / (t * bsj.server.max_cap)
                    diff = min_beta - beta[a.name][bsj.name]
                    if beta[a.name][bsj.name] < min_beta and diff < remaining:
                        beta[a.name][bsj.name] += diff
                        remaining -= diff
                    elif beta[a.name][bsj.name] < min_beta:
                        beta[a.name][bsj.name] += remaining
                        remaining = 0
                if remaining > 0:
                    for a in services:
                        beta[a.name][bsj.name] += remaining / len(services)

        return beta

    def _check_results(
        self,
        lmbda: Dict[str, Dict[str, int]],
        gamma: Dict[str, Dict[str, Dict[str, float]]],
        beta: Dict[str, Dict[str, float]],
        infr: Infrastructure,
    ):
        """Checks aditional constraints to see that results are valid.
        Args:
            lmbda (Dict[str, Dict[str, int]]): Requests per service and bs.
            gamma (Dict[str, Dict[str, Dict[str, float]]]): Requests routing.
            beta (Dict[str, Dict[str, float]]): Computing resource allocation.
            infr (Infrastructure): The infrastructure.
        """
        # Links cannot produce new requests (gamma cannot be negative)
        if any(
            gamma[bsi][ak][bsj] < 0
            for bsi in gamma
            for ak in gamma[bsi]
            for bsj in gamma[bsi][ak]
        ):
            raise Exception("Gamma cannot be negative.")

        if any(
            beta[ak][bsj] < 0
            for ak in beta
            for bsj in beta[ak]
        ):
            raise Exception("Beta cannot be negative.")

        # Check that all requests are routed
        if any(
            round(sum(gamma[bsi][ak].values()), 3) < 1 and lmbda[bsi][ak] != 0
            for bsi, ak in itertools.product(infr.bss, infr.services)
        ):
            for bsi, ak in itertools.product(infr.bss, infr.services):
                if (
                    round(sum(gamma[bsi][ak].values()), 3) < 1
                    and lmbda[bsi][ak] != 0
                ):
                    raise Exception(
                        f"Some requests received in {bsi} are not routed. Total gamma is {round(sum(gamma[bsi][ak].values()), 3)}. Total requests is {lmbda[bsi][ak]}."
                    )

            raise Exception("Not all requests are routed")

        # Check that requests are not routed more than once
        if any(
            round(sum(gamma[bsi][ak].values()), 2) > 1
            for bsi, ak in itertools.product(infr.bss, infr.services)
        ):
            raise Exception("gamma cannot be bigger than 1.")

        # Check that servers are not overloaded
        for bsj in infr.bss:
            if (
                round(
                    sum(beta[ak][bsj] for ak in beta),
                    2,
                )
                > 1
            ):
                raise Exception(
                    f"Server {bsj} overload. Total load {sum(beta[ak][bsj] for ak in beta)}"
                )

    def threshold_based(
        self, lmbda: Dict[str, Dict[str, float]], threshold: float
    ):
        """Calculates gamma, beta and eta based on a predifined threshold. If the load
        of a server is below the threshold it is turned off.
        Args:
            lmbda (Dict[str, Dict[str, float]]): Requests per service and bs.
            threshold (float): The threshold.
        Returns
            Dict[str, Dict[str, Dict[str, float]]]: The gamma matrix.
            Dict[str, Dict[str, Dict[str, float]]]: The beta matrix.
            Dict[str, Dict[str, float]]: The eta matrix.
        """
        infr = self.infr
        eta, gamma, beta, used_cap = self.initialize(lmbda, self.infr)

        for bsj in self.infr.bss.values():
            if bsj.server is None:
                continue
            utilization = (
                sum(
                    self._req_to_ops(
                        gamma[bs][a.name][bsj.name] * lmbda[bs][a.name], a
                    )
                    for bs in infr.bss
                    for a in infr.services.values()
                )
                / bsj.server.max_cap
            )
            if (
                bsj.server is not None
                and utilization < threshold
                and sum(eta[bs] for bs in eta) > 1
            ):
                # Turn off the server
                eta[bsj.name] = 0
                for a, bsi in itertools.product(
                    infr.services.values(), infr.bss.values()
                ):
                    # Requests to this service coming from this bs that need relocation
                    req_to_reroute = math.ceil(
                        lmbda[bsi.name][a.name]
                        * gamma[bsi.name][a.name][bsj.name]
                    )
                    gamma, req_to_reroute, used_cap, _ = self._reroute(
                        [
                            bs
                            for bs in infr.bss.values()
                            if bs.server is not None
                            and eta[bs.name] != 0
                            and bs.name != bsj.name
                        ],
                        bsi,
                        a,
                        req_to_reroute,
                        lmbda,
                        gamma,
                        used_cap,
                    )
                    # The off should not have requests routed to it
                    gamma[bsi.name][a.name][bsj.name] = 0
                    used_cap[bsj.name] = 0

        beta = self._calculate_beta_matrix(lmbda, gamma, infr)

        return eta, gamma, beta
