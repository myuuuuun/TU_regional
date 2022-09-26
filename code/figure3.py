import numpy as np
import pandas as pd
import datetime
import cvxpy as cvx
np.set_printoptions(precision=3)
np.set_printoptions(suppress=False)


def G(nx_list, U):
    U_0 = cvx.hstack([U, np.zeros((len(nx_list), 1))])
    return nx_list @ cvx.log_sum_exp(U_0, axis=1)


def H(my_list, V):
    V_0 = cvx.vstack([V, np.zeros((1, len(my_list)))])
    return my_list @ cvx.log_sum_exp(V_0, axis=0)


def obj_fun_EAE(U, V, bar_w, underline_w, nx_list, my_list, bar_o, underline_o):
    # G(U), H(V)
    value = G(nx_list, U) + H(my_list, V)

    # \bar{w}_z \bar{o}_z
    value += cvx.sum(bar_w @ bar_o)

    # \underline{w}_z \underline{o}_z
    value -= cvx.sum(underline_w @ underline_o)

    return value


def constraints_EAE(U, V, bar_w, underline_w, region_list):
    w = (bar_w - underline_w)[region_list]
    val = U + V + cvx.vstack([w for _ in range(U.shape[0])])

    return val


def compute_EAE(
    phi, 
    nx_list, 
    my_list, 
    region_list, 
    bar_o, 
    underline_o, 
    init=None, 
    tol=1e-8, 
    maxiter=100
    ):
    x_size = len(nx_list)
    y_size = len(my_list)
    z_size = len(bar_o)

    U = cvx.Variable((x_size, y_size))
    V = cvx.Variable((x_size, y_size))
    bar_w = cvx.Variable(z_size)
    underline_w = cvx.Variable(z_size)

    obj = cvx.Minimize(obj_fun_EAE(U, V, bar_w, underline_w, nx_list, my_list, bar_o, underline_o))
    constraints = [
        constraints_EAE(U, V, bar_w, underline_w, region_list) == phi, 
        bar_w >= 0, 
        underline_w >= 0
    ]

    prob = cvx.Problem(obj, constraints)
    prob.solve(qcp=True, verbose=True)
    #print(U.value)
    #print(V.value)
    #print(bar_w.value)
    #print(underline_w.value)


def simulate(x_size, y_size, z_size, rng):
    x_total_pop, y_total_pop = 1.0, 1.5
    x_pop_unit = x_total_pop / x_size
    y_pop_unit = y_total_pop / y_size
    lb_z = y_total_pop / 5 / z_size

    nx_list = np.repeat(x_pop_unit, x_size)
    my_list = np.repeat(y_pop_unit, y_size)
    region_list = np.repeat(np.arange(z_size), repeats=(y_size//z_size))

    bar_oz = np.full([z_size], x_total_pop)
    underline_oz = np.full([z_size], lb_z)
    phi = rng.normal(loc=2.0, scale=1.0, size=[x_size, y_size])

    compute_EAE(
        phi, nx_list, my_list, region_list, bar_oz, underline_oz, 
        tol=1e-3, maxiter=10000
    )


class IndEq(object):
    """
    Individual equilibrium
    """
    def __init__(self, i_size, j_size, region_list):
        self.i_size = i_size
        self.j_size = j_size
        self.region_list = region_list


    def obj_fun_IE(self, u, v):
        return cvx.sum(u) + cvx.sum(v)


    def constraints_IE(self, u, v, phi_w):
        umat = cvx.reshape(cvx.hstack([u for _ in range(self.j_size)]), (self.i_size, self.j_size))
        vmat = cvx.vstack([v for _ in range(self.i_size)])
        val = umat + vmat - phi_w
        
        return val


    def compute_IE_given_w(self, phi, w):
        u = cvx.Variable(self.i_size)
        v = cvx.Variable(self.j_size)

        phi_w = phi - (w[self.region_list])[None, :]

        obj = cvx.Minimize(self.obj_fun_IE(u, v))
        constraints = [
            self.constraints_IE(u, v, phi_w) >= 0,
            u >= 0,
            v >= 0
        ]

        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True)

        print(prob.value, u.value, v.value, prob.constraints[0].dual_value)


if __name__ == "__main__":
    """
    sample_size = 1
    num_simulation = 10
    rng = np.random.default_rng(seed=2021)
    time_results = []

    for x_size in [10, 20]:
        for z_size in range(5, 101, 5):
            y_size = 10 * z_size
            start = datetime.datetime.now()

            for _ in range(num_simulation):
                simulate(x_size, y_size, z_size, rng)
            
            time_results.append({
                "x": x_size,
                "y": y_size,
                "z": z_size,
                "total_seconds": (datetime.datetime.now() - start).total_seconds() / num_simulation
            })

    result_df = pd.DataFrame(time_results)
    print(result_df)
    result_df.to_csv("performance_results.csv")
    """

    """
    phi = np.arange(6).reshape([2, 3])
    region_list = np.array([0, 1, 1])
    w = np.array([0.1, 0.2])

    market = IndEq(2, 3, region_list)
    market.compute_IE_given_w(phi, w)
    """

    # Generate a random non-trivial linear program.
    m = 15
    n = 10
    np.random.seed(1)
    s0 = np.random.randn(m)
    lamb0 = np.maximum(-s0, 0)
    s0 = np.maximum(s0, 0)
    x0 = np.random.randn(n)
    A = np.random.randn(m, n)
    b = A @ x0 + s0
    c = -A.T @ lamb0

    # Define and solve the CVXPY problem.
    x = cvx.Variable(n)
    prob = cvx.Problem(cvx.Minimize(c.T@x),
                     [A @ x <= b])
    prob.solve(verbose=True)

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution is")
    print(prob.constraints[0].dual_value)

