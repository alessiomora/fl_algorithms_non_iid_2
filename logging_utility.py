import itertools as it
import os

PATH = "emnist_100_results"


def write_logs(file_name, history, folder, client=None):
    folder = os.path.join(PATH, folder)
    prefix = ""
    if client is not None:
        prefix = "[" + str(client) + "]" + " "

    f = open(os.path.join(folder, file_name), "a")
    f.write(prefix + str(history))
    f.write("\n")
    f.close()
    return


def write_intro_logs(list_file_names, dict_sett, folder):
    folder = os.path.join(PATH, folder)

    exist = os.path.exists(folder)

    if not exist:
        os.makedirs(folder)

    for file_name in list_file_names:
        f = open(os.path.join(folder, file_name), "a")
        f.write("Simulation Start " + str(dict_sett))
        f.write("\n")
        f.close()
    return


def get_all_hp_combinations(hp):
    '''Turns a dict of lists into a list of dicts'''
    combinations = it.product(*(hp[name] for name in hp))
    # for value in combinations:
    #     print(value)
    hp_dicts = [{key: value[i] for i, key in enumerate(hp)} for value in combinations]
    return hp_dicts

def encode_general_hyperparameter_in_string(s):
    encoded = ""

    for h in s.keys():
        if h not in ["algorithm", 'reinit_classifier', 'batch_size', 'rounds', 'dataset', "alpha", "seed",
                    "architecture", "total_clients", "E", "C",
                     "mu", "gamma", "lambda1_lambda2_", "beta", "M", "alpha_dyn",
                     "aggregation", "distill_batch_size", "max_distill_epochs_server",
                     "max_distill_epochs_clients", "distill_examples"]:
            if encoded == "":
                if not isinstance(s[h], str) and not isinstance(s[h], list):
                    encoded = encoded + h + str(round(s[h], 5))
                elif isinstance(s[h], list):
                    encoded = encoded + h + str(s[h])
                else:
                    encoded = encoded + h + s[h]
            else:
                if not isinstance(s[h], str) and not isinstance(s[h], list):
                    encoded = encoded + "_" + h + str(round(s[h], 5))
                elif isinstance(s[h], list):
                    encoded = encoded + "_" + h + str(s[h])
                else:
                    encoded = encoded + "_" + h + s[h]

    return encoded.replace(" ", "")

def encode_specific_hyperparameter_in_string(s):
    encoded = ""

    for h in s.keys():
        if h in ["beta", "mu", "gamma", "lambda1_lambda2_", "M", "alpha_dyn"]:
            if encoded == "":
                if not isinstance(s[h], str) and not isinstance(s[h], list):
                    encoded = encoded + h + str(round(s[h], 5))
                elif isinstance(s[h], list):
                    encoded = encoded + h + str(s[h])
                else:
                    encoded = encoded + h + s[h]
            else:
                if not isinstance(s[h], str) and not isinstance(s[h], list):
                    encoded = encoded + "_" + h + str(round(s[h], 5))
                elif isinstance(s[h], list):
                    encoded = encoded + "_" + h + str(s[h])
                else:
                    encoded = encoded + "_" + h + s[h]

    return encoded.replace(" ", "")

def encode_hyperparameter_in_string(s):
    encoded = ""

    for h in s.keys():
        if h not in ["algorithm", 'reinit_classifier', 'batch_size', 'rounds', 'dataset', "alpha", "seed", "norm"]:
            if encoded == "":
                if not isinstance(s[h], str) and not isinstance(s[h], list):
                    encoded = encoded + h + str(round(s[h], 5))
                elif isinstance(s[h], list):
                    encoded = encoded + h + str(s[h])
                else:
                    encoded = encoded + h + s[h]
            else:
                if not isinstance(s[h], str) and not isinstance(s[h], list):
                    encoded = encoded + "_" + h + str(round(s[h], 5))
                elif isinstance(s[h], list):
                    encoded = encoded + "_" + h + str(s[h])
                else:
                    encoded = encoded + "_" + h + s[h]

    if "seed" in s.keys():
        encoded = encoded + ",seed" + str(round(s["seed"], 3))

    return encoded.replace(" ", "")


def get_combinations(hp, cd):
    settings = get_all_hp_combinations(hp)
    configs = []
    for s in settings:
        if not s["algorithm"] in cd.keys():
            configs.append(s)
        if s["algorithm"] in cd.keys():
            combinations = get_all_hp_combinations(cd[s["algorithm"]])
            for c in combinations:
                new_setting = s.copy()
                new_setting.update(c)
                configs.append(new_setting)

    return configs

