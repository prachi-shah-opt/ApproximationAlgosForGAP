import numpy as np
import gurobipy as gp
from gurobipy import LinExpr, Column


class GAPProblem:

    def __init__(self, filename, env=None):
        with open(filename, "r") as f:
            lines = [list(map(int, l[:-1].split(" "))) for l in f.readlines()]

        self.num_machines, self.num_jobs, self.optimal_obj = lines[0]
        self.cost = np.array(lines[1: 1 + self.num_machines], dtype=np.uint8)
        self.proc_time = np.array(lines[1 + self.num_machines: 1 + 2 * self.num_machines], dtype=np.uint8)
        self.capacity = np.array(lines[-1], dtype=np.uint8)

        self.machines_list = list(range(self.num_machines))
        self.jobs_list = list(range(self.num_jobs))

        self.algorithm = None
        self.algo_functions = {
            "compact-lp-assignment": self.compact_lp_assignment,
            "shmyos-tardos": self.shmoys_tardos,
            "iterative-lp": self.iterative_lp,
            "local-search": self.local_search,
            "config-lp-rounding": self.config_lp_rounding,
        }

        if env is None:
            self.gp_env = gp.Env()
            self.gp_env.setParam("OutputFlag", 0)
        else:
            self.gp_env = env

    def solve(self):
        return self.algo_functions[self.algorithm]()

    def set_algorithm(self, algo):
        self.algorithm = algo

    def create_lp_model(self):
        lp_model = gp.Model(env=self.gp_env)
        lp_model.ModelSense = -1

        assign_vars = lp_model.addVars(self.num_machines, self.num_jobs, obj=self.cost, name='x')

        for each in np.argwhere(self.proc_time > self.capacity[:, None]):
            assign_vars[each[0], each[1]].ub = 0

        job_constrs = lp_model.addConstrs(assign_vars.sum('*', j) <= 1 for j in self.jobs_list)
        machine_constrs = lp_model.addConstrs(
            LinExpr(self.proc_time[i, :], assign_vars.select(i, '*')) <= self.capacity[i]
            for i in self.machines_list)

        return lp_model, assign_vars, job_constrs, machine_constrs

    def get_lp_solution(self):
        lp_model, assign_vars, _, _ = self.create_lp_model()
        lp_model.optimize()

        init_cols = [[] for _ in self.machines_list]
        for (i, j), var in assign_vars.items():
            if var.x >= 1 - 1e-6:
                init_cols[i].append(j)

        return init_cols

    def shmoys_tardos(self):

        lp_model, assign_vars, _, _ = self.create_lp_model()
        lp_model.optimize()

        bipartite_edges = []  # (job, machine, slot)
        machine_slots = np.empty(self.num_machines, dtype=np.uint8)
        for i in self.machines_list:
            p_jobs = sorted(((self.proc_time[i, j], j) for j in self.jobs_list if assign_vars[i, j].x > 0),
                            reverse=True)

            curr_slot, remaining_cap = 0, 1.0
            for _, j in p_jobs:
                bipartite_edges.append((j, i, curr_slot))
                remaining_cap -= assign_vars[i, j].x
                if remaining_cap < 1e-6:  # capacity = 0
                    curr_slot += 1
                    remaining_cap += 1.0
                    if remaining_cap < 1 - 1e-6:  # spillover
                        bipartite_edges.append((j, i, curr_slot))
            machine_slots[i] = curr_slot + int(remaining_cap < 1-1e-6)

        bipartite_model = gp.Model(env=self.gp_env)
        bipartite_model.ModelSense = -1

        y = bipartite_model.addVars(bipartite_edges,
                                    obj=[self.cost[i, j] for (j, i, _) in bipartite_edges],
                                    name="y")
        for i in self.machines_list:
            for k in range(machine_slots[i]):
                bipartite_model.addConstr(y.sum('*', i, k) <= 1, name="machine_node")

        for j in self.jobs_list:
            bipartite_model.addConstr(y.sum(j, "*", "*") == 1, name="job_node")

        bipartite_model.optimize()

        solution = {}
        edge_num = 0
        total_cost = 0

        for i in self.machines_list:
            option0 = None
            option_set = []

            edge = bipartite_edges[edge_num]
            while edge[1] == i and edge[2] == 0 and edge_num < len(bipartite_edges):
                if y[edge].x > 0:
                    assert y[edge].x > 1 - 1e-6
                    option0 = edge[0]

                edge_num += 1
                try:
                    edge = bipartite_edges[edge_num]
                except IndexError:
                    pass

            while edge[1] == i and edge_num < len(bipartite_edges):
                if y[edge].x > 0:
                    assert y[edge].x > 1 - 1e-6
                    option_set.append(edge[0])

                edge_num += 1
                try:
                    edge = bipartite_edges[edge_num]
                except IndexError:
                    pass

            if option0 is None and option_set == []:
                solution[i] = []
                continue

            option0_time = self.proc_time[i, option0].item() if option0 is not None else 0
            option_set_time = self.proc_time[i, option_set].sum()
            assert option0_time <= self.capacity[i]
            assert option_set_time <= self.capacity[i]

            option0_cost = self.cost[i, option0].item() if option0 is not None else 0
            option_set_cost = self.cost[i, option_set].sum()

            if option0_time + option_set_time <= self.capacity[i]:
                solution[i] = option_set + [option0, ]
                total_cost += option0_cost + option_set_cost

            elif option_set_cost >= option0_cost:
                solution[i] = option_set
                total_cost += option_set_cost
            else:
                solution[i] = [option0, ]
                total_cost += option0_cost

        return total_cost

    def iterative_lp(self):
        lp_model, assign_vars, job_constrs, machine_constrs = self.create_lp_model()

        assigned = np.full(self.num_jobs, -1, np.int8)
        relaxed = np.full(self.num_machines, 0, np.int)
        overflow_assignments = {i: [] for i in self.machines_list}
        assignments = {i: [] for i in self.machines_list}

        deg_machine = np.empty(self.num_machines, dtype=np.uint8)
        delta_machine = np.empty(self.num_machines, dtype=np.float)

        while np.any(assigned == -1):
            deg_machine[:] = 0
            delta_machine[:] = 0
            lp_model.optimize()

            reduce_capacity = np.zeros(self.num_machines, dtype=np.uint8)
            for i, j in assign_vars.keys():
                if i == 8:
                    print(end='')
                val = assign_vars[i, j].x
                if val <= 1e-6:
                    lp_model.remove(assign_vars[i, j])
                    del assign_vars[(i, j)]

                elif val >= 1 - 1e-6:
                    lp_model.remove(assign_vars[i, j])
                    lp_model.remove(job_constrs[j])
                    del assign_vars[(i, j)]
                    del job_constrs[j]

                    assigned[j] = i
                    if relaxed[i] == 0:
                        reduce_capacity[i] += self.proc_time[i, j]
                        assignments[i].append(j)
                    else:
                        overflow_assignments[i].append(j)

                else:
                    deg_machine[i] += 1
                    delta_machine[i] += val

            for i in np.argwhere(reduce_capacity > 0).flatten():
                machine_constrs[i].rhs = machine_constrs[i].rhs - reduce_capacity[i]

            for i in np.argwhere(
                    (relaxed == 0) & ((deg_machine == 1) | ((deg_machine == 2) & (delta_machine >= 1)))).flatten():
                lp_model.remove(machine_constrs[i])
                del machine_constrs[i]
                relaxed[i] = 1

        solution = {}
        total_cost = 0
        for i in self.machines_list:
            if len(overflow_assignments[i]) == 0:
                solution[i] = assignments[i]
                total_cost += self.cost[i, solution[i]].sum()

            elif len(overflow_assignments[i]) == 2:
                j1, j2 = overflow_assignments[i]
                if self.proc_time[i, j1] <= self.proc_time[i, j2]:
                    assignments[i].append(j1)
                    overflow_assignments[i] = [j2, ]
                else:
                    assignments[i].append(j2)
                    overflow_assignments[i] = [j1, ]

            if len(overflow_assignments[i]) == 1:
                option0 = overflow_assignments[i]
                option_set = assignments[i]

                option0_time = self.proc_time[i, option0].item()
                option_set_time = self.proc_time[i, option_set].sum()
                assert option0_time <= self.capacity[i]
                assert option_set_time <= self.capacity[i]

                option_set_cost = self.cost[i, option_set].sum()
                option0_cost = self.cost[i, option0].item()

                if option0_time + option_set_time <= self.capacity[i]:
                    solution[i] = option_set + [option0, ]
                    total_cost += option_set_cost + option0_cost
                elif option_set_cost >= option0_cost:
                    solution[i] = option_set
                    total_cost += option_set_cost
                else:
                    solution[i] = [option0, ]
                    total_cost += option0_cost

        return total_cost

    def config_lp_rounding(self):

        init_cols = self.get_lp_solution()

        rmp = gp.Model(env=self.gp_env)
        rmp.ModelSense = -1

        init_vars = rmp.addVars(self.num_machines, obj=[self.cost[i, init_cols[i]].sum() for i in self.machines_list],
                                name=[f"v_{i}_0" for i in self.machines_list])
        job_cons = rmp.addConstrs((LinExpr(0) <= 1) for _ in self.jobs_list)
        machine_cons = rmp.addConstrs((init_vars[i] == 1) for i in self.machines_list)

        # initialize restricted master problem
        for i in self.machines_list:
            init_jobs = init_cols[i]
            for j in init_jobs:
                rmp.chgCoeff(job_cons[j], init_vars[i], 1)

        # pricing_models = [gp.Model(env=self.gp_env) for _ in self.machines_list]
        # pricing_vars = []
        #
        # # initialize pricing subproblems
        # for i in self.machines_list:
        #     model = pricing_models[i]
        #     model.ModelSense = -1
        #
        #     vars = model.addVars(self.num_jobs, vtype='B')
        #     pricing_vars.append(vars)
        #
        #     model.addConstr(LinExpr(self.proc_time[i, :], vars.values()) <= self.capacity[i], "knapsack")

        # solve column generation here
        new_cols_added = True
        iters = 0
        num_cols = np.full(self.num_machines, 1, dtype=int)
        while new_cols_added:
            iters += 1

            new_cols_added = False
            rmp.optimize()

            job_duals = np.array([job_cons[j].Pi for j in self.jobs_list])
            machine_duals = np.array([machine_cons[i].Pi for i in self.machines_list])

            for i in self.machines_list:
                # model = pricing_models[i]
                # vars = pricing_vars[i]
                #
                # model.setObjective(LinExpr(self.cost[i, :] - job_duals, vars.values()))
                # model.optimize()
                # model.write(f"knapsack_{i}_{iters}.lp")
                #
                # if model.objVal > machine_duals[i] + 1e-6:
                #     new_col = [j for j in self.jobs_list if vars[j].x == 1]
                #     rmp.addVar(obj=self.cost[i, new_col].sum(),
                #                column=Column(
                #                    [1 for _ in range(len(new_col) + 1)],
                #                    [job_cons[j] for j in new_col] + [machine_cons[i], ]),
                #                name=f"v_{i}_{num_cols[i]}")
                #     new_cols_added = True

                new_col, solution_value = self.solve_knapsack(self.cost[i, :] - job_duals, self.proc_time[i, :], self.capacity[i])
                if solution_value > machine_duals[i] + 1e-6:
                    rmp.addVar(obj=self.cost[i, new_col].sum(),
                               column=Column(
                                   [1 for _ in range(len(new_col) + 1)],
                                   [job_cons[j] for j in new_col] + [machine_cons[i], ]),
                               name=f"v_{i}_{num_cols[i]}")
                    new_cols_added = True


        cols_in_sol = [v for v in rmp.getVars() if v.x > 1e-6]
        u = np.random.random(self.num_machines)
        cumulative_x = np.zeros_like(u)
        cols_selected = [None, ] * self.num_machines

        for v in cols_in_sol:
            i = int(v.varname.split("_")[1])

            cumulative_x[i] += v.x
            if cumulative_x[i] > u[i]:
                cols_selected[i] = v
                u[i] = 1

        assert np.all(np.abs(cumulative_x - 1) < 1e-6)

        jobs_assignment = np.full(self.num_jobs, -1, dtype=np.int8)

        for i in self.machines_list:
            for j in self.jobs_list:
                if rmp.getCoeff(job_cons[j], cols_selected[i]) == 1:
                    if jobs_assignment[j] == -1 or self.cost[i, j] > self.cost[jobs_assignment[j], j]:
                        jobs_assignment[j] = i

        solution = {i: np.argwhere(jobs_assignment == i).flatten().tolist() for i in self.machines_list}
        total_cost = sum(self.cost[i, solution[i]].sum() for i in self.machines_list)

        return total_cost

    def compact_lp_assignment(self):
        solution_cols = self.get_lp_solution()
        solution = {i: solution_cols[i] for i in self.machines_list}
        total_cost = sum(self.cost[i, solution[i]].sum() for i in self.machines_list)

        return total_cost

    def local_search(self, epsilon=1e-3):
        machine_cost = np.zeros(self.num_machines)

        prospects = [[] for _ in self.machines_list]
        deltas = np.zeros(self.num_machines)
        job_assignments = np.full(self.num_jobs, -1, dtype=np.int)

        for iters in range(int(self.num_machines * np.log(1/epsilon))):

            for i in self.machines_list:
                curr_value = np.zeros(self.num_jobs, dtype=np.int)
                for j in self.jobs_list:
                    if job_assignments[j] not in [-1, i]:
                        curr_value[j] = self.cost[job_assignments[j], j]
                marginal_cost = np.subtract(self.cost[i, :], curr_value, casting='safe')
                prospects[i], deltas[i] = self.solve_knapsack(marginal_cost, self.proc_time[i, :], self.capacity[i])

            deltas -= machine_cost

            i_star = np.argmax(deltas)
            if deltas[i_star] <= 0:
                break

            job_assignments[job_assignments == i_star] = -1
            job_assignments[prospects[i_star]] = i_star

            for i in self.machines_list:
                machine_cost[i] = np.sum(self.cost[i, (job_assignments==i)])

        solution = {i: np.argwhere(job_assignments == i).flatten().tolist() for i in self.machines_list}
        total_cost = sum(self.cost[i, solution[i]].sum() for i in self.machines_list)

        assert total_cost == machine_cost.sum()
        return total_cost

    def solve_knapsack(self, profits, weights, capacity):
        demands = weights
        solution_value = 0.0
        solution = []

        myweights = []
        myitems = []
        myprofits = []

        mycapacity = capacity

        # Remove unnecessary items
        for j in range(self.num_jobs):
            if profits[j] <= 0.0:
                pass
            else:
                myweights.append(demands[j])
                myprofits.append(profits[j])
                myitems.append(j)

        n = len(myitems)
        myweights = np.array(myweights)
        myprofits = np.array(myprofits)

        dp = np.zeros((n+1, mycapacity + 1))

        for i in range(1, n+1):
            for w in range(mycapacity + 1):
                if myweights[i - 1] <= w:
                    dp[i, w] = max(dp[i - 1, w], myprofits[i - 1] + dp[i - 1, w - myweights[i - 1]])
                else:
                    dp[i, w] = dp[i - 1, w]

        # Retrieve the optimal solution
        i, j = n, mycapacity
        while i > 0 and j > 0:
            if dp[i, j] != dp[i - 1, j]:
                solution.append(myitems[i - 1])
                j -= myweights[i - 1]
            i -= 1
        solution_value += dp[-1, -1]

        return solution, solution_value
