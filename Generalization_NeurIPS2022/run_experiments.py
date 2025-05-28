import warnings
import time
import argparse
import numpy as np
import pandas as pd
from copy import deepcopy

from utils.dataloaders import load_binary_data, make_classification_data
from utils.models import federated_learning, centralized_learning

warnings.filterwarnings("ignore")




def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--save_path", type=str, default="save/")
    parser.add_argument("--name", type=str, default="mnist", help="dataset name")
    parser.add_argument("--comparison", type=str, default="K", choices=["rho", "n", "K"],
                        help="to run comparison for different 'rho' or 'n'.")
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--K_values", nargs="+", type=int, default=[2, 5, 10, 25, 50])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--n_values", nargs="+", type=float, default=[100, 200, 400, 500, 700, 900, 1000])
    parser.add_argument("--noise_std", type=float, default=0.2,
                        help="standard deviation of the AWGN/Gaussian data")
    parser.add_argument("--dim", type=int, default=100, help="dimension of synthetic data")
    parser.add_argument("--radius", type=float, default=1.5, help="radius of balls (synthetic data)")
    parser.add_argument("--rho_values", nargs="+", type=float, default=[1.0, 2.0, 3.0, 4.0, 5.0])
    parser.add_argument("--classes", nargs="+", type=int, default=[1,6],
                        help="MNIST classes")
    parser.add_argument("--compare_hom_het", type=int, default=0, 
                        help="whether to compare the homogeneous and heterogeneous setups (NIPS25 paper)")
    
    parser.add_argument("--loss", type=str, default="hinge")
    parser.add_argument("--rounds", type=int, default=1, help="number of communication rounds")
    parser.add_argument("--proj_dim", type=int, default=0, help="SVM Gaussian kernel dimension")
    parser.add_argument("--gamma", type=float, default=0.05, help="SVM Gaussian kernel parameter")
    parser.add_argument("--client_epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs for centralized training")
    parser.add_argument("--lr", type=float, default=0.01)                    
    parser.add_argument("--MC", type=int, default=10,
                        help="number of runs (Monte-Carlo simulations)")
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    return args



def run_comparison(args):
    params = deepcopy(args)
    params.N = params.K*params.n # N = nK
    params.task = "classification"

    comp = params.comparison+"_values"
    values = getattr(params, comp)
    # if args.comparison == "n":
    #     values = args.n_values
    # elif args.comparison == "rho":
    #     values = args.rho_values

    mnist = None
    if params.name == "mnist":
        # Preload whole MNIST dataset to avoid reloading everytime
        mnist = load_binary_data(params.classes[0], params.classes[1], params.data_path, params.proj_dim, params.gamma)

    df = pd.DataFrame(0, 
                      index=values, 
                    #   columns=["emp_01", "risk_01", "emp", "risk"])
                      columns=["dis_emp", "dis_risk", "emp", "risk", 
                               "dis_emp_std", "dis_risk_std", "emp_std", "risk_std", "bias", "bias_std"])
    
    for v in values:
        if params.comparison == "n":
            params.N = params.K*v
        elif params.comparison == "K":
            params.N = params.n*v
        elif params.comparison == "rho":
            params.radius = v
        print(f"{params.comparison} = {v} \n")

        # emp_risks_01, risks_01 = [], []
        dis_emp_risks, dis_risks = [], []
        emp_risks, risks = [], []
        biases = []
        # fed_times, times = [], []

        for m in range(params.MC):
            print("m =", m+1)
            params.seed = args.seed + m

            data = make_classification_data(params, mnist)

            start = time.time()
            server = federated_learning(data, params, params.K)
            end = time.time()
            # fed_times.append(end-start)

            dis_emp_01, dis_risk_01 = server.compute_01_risks()
            dis_emp, dis_risk = server.compute_emp_risk(), server.compute_test_risk()
            biases.append(dis_emp - server.avg_local_emp_risk)

            print("Federated learning setup total runtime: {0:.3f}s.".format(end-start))
            print("0-1 Empirical risk/Test risk: {0:.3f}/{1:.3f}".format(dis_emp_01, dis_risk_01))
            print("Empirical risk: {0:.3f}".format(dis_emp))
            print("==============================")
            
            start = time.time()
            # emp, risk, emp_01, risk_01 = centralized_learning(data, params)
            server = federated_learning(data, params, K=1)
            end = time.time()
            # times.append(end-start)

            emp_01, risk_01 = server.compute_01_risks()
            emp, risk = server.compute_emp_risk(), server.compute_test_risk()

            print("Centralized learning setup total runtime: {0:.3f}s.".format(end-start))
            print("0-1 Empirical risk/Test risk: {0:.3f}/{1:.3f}".format(emp_01, risk_01))
            print("Empirical risk: {0:.3f}".format(emp))
            
            # emp_risks_01.append(emp_01)
            # risks_01.append(risk_01)
            dis_emp_risks.append(dis_emp)
            dis_risks.append(dis_risk)
            emp_risks.append(emp)
            risks.append(risk)
            print("=============================")
        print("===============================================")
        df.loc[v] = [np.mean(dis_emp_risks), np.mean(dis_risks), np.mean(emp_risks), np.mean(risks),
                    np.std(dis_emp_risks), np.std(dis_risks), np.std(emp_risks), np.std(risks),
                    np.mean(biases), np.std(biases)]
    # df["dis_emp"].loc[v] = np.mean(dis_emp_risks)
    # df["dis_risk"].loc[v] = np.mean(dis_risks)
    # df["emp"].loc[v] = np.mean(emp_risks)
    # df["risk"].loc[v] = np.mean(risks)
    # df["bias"].loc[v] = np.mean(biases)

    # df["dis_emp_std"].loc[v] = np.std(dis_emp_risks)
    # df["dis_risk_std"].loc[v] = np.std(dis_risks)
    # df["emp_std"].loc[v] = np.std(emp_risks)
    # df["risk_std"].loc[v] = np.std(risks)
    # df["bias_std"].loc[v] = np.std(biases)

    # df["gen_01"] = df["risk_01"] - df["emp_01"]
    # df["gen"] = df["risk"] - df["emp"]

    return df


def main(args):
    if args.comparison == "n":
        param = 'rho'
        val = args.radius
    elif args.comparison == 'K':
        param = 'n'
        val = args.n
    elif args.comparison == "rho":
        param = 'n'
        val = args.n

    if args.compare_hom_het:
        args.iid = True
        df = run_comparison(args)
        df.to_pickle(args.save_path+f"comp_{args.comparison}/values_iid_{args.name}_d{args.dim}_lr{args.lr}_e{args.client_epochs}_{param}{val}_MC{args.MC}.pickle")
        
        args.iid = False
        df = run_comparison(args)
        df.to_pickle(args.save_path+f"comp_{args.comparison}/values_noniid_{args.name}_d{args.dim}_lr{args.lr}_e{args.client_epochs}_{param}{val}_MC{args.MC}.pickle")

    else:
        df = run_comparison(args)
        data_info = args.name
        if data_info == "mnist":
            data_info += f"_{args.classes}_gamma{args.gamma}_dim{args.proj_dim}"
        df.to_pickle(args.save_path+f"comp_{args.comparison}/values_{data_info}_lr{args.lr}_e{args.client_epochs}_{param}{val}_MC{args.MC}.pickle")

    return df




if __name__ == '__main__':
    args = args_parser()

    main(args)