import random
import math
import sys
import os
import json
import copy

app_task_count = 0
sim_time = 100
templates_dir = "./templates/"
execution_dir = "./execution/"
json_ext = ".json"
execution_file_break_char = "___"



# def get_last_layer_size(inner_layers):
#     if len(inner_layers) != 0:
#         return len(inner_layers[len(inner_layers) - 1])
#     return 0


# def retrieve_options(layer_list, last_layer_size, input_layer_size):
#     number_links = 1
#     layer_list_size = len(layer_list)

#     if last_layer_size > 1:
#         number_links = random.randrange(1, last_layer_size)

#     options = random.sample(range(last_layer_size), number_links)
#     total_index_size = input_layer_size
#     for j in range(0, layer_list_size - 1):
#         total_index_size = total_index_size + len(layer_list[j])

#     options = list(map(lambda option: option + total_index_size, options))
#     return options


# def generate_applications(max_layer_size, count):
#     global app_task_count

#     app_task_count = 0

#     input_layer_size = 1
#     output_layer_size = 1
#     inner_layer_count = random.randrange(1, max_layer_size)

#     input_layer = []
#     inner_layers = []
#     output_layer = []
#     res = []

#     # GENERATING INPUT LAYER
#     for i in range(input_layer_size):
#         input_layer.append((get_task_perm(f'APP_{count}_Input_Layer_{i}'), [], []))

#     # GENERATING INNER LAYERS
#     for i in range(inner_layer_count):
#         layer = []
#         layer_size = random.randrange(1, max_layer_size)

#         for x in range(layer_size):
#             task = get_task_perm(f'APP_{count}_Inner_Layer_{i}:{x}')

#             last_layer_size = get_last_layer_size(inner_layers)

#             in_edges = []
#             out_edges = []

#             layer.append((task, in_edges, out_edges))
#         inner_layers.append(layer)

#     # GENERATING OUTPUT LAYER
#     for i in range(0, output_layer_size):
#         task = get_task_perm(f'APP_{count}_Output_Layer_{i}')

#         last_layer_size = get_last_layer_size(inner_layers)

#         in_edges = []
#         out_edges = []

#         output_layer.append((task, in_edges, out_edges))

#     for i in range(0, len(inner_layers)):
#         for x in range(0, len(inner_layers[i])):
#             if(i == 0):
#                 input_layer[0][2].append(inner_layers[i][x][0]["id"])
#                 inner_layers[i][x][1].append(input_layer[0][0]["id"])

#             if len(inner_layers[i][x][1]) == 0:
#                 in_edge_count = 1
                
#                 if len(inner_layers[i - 1]) > 1:
#                     in_edge_count = random.randrange(1, len(inner_layers[i - 1]))
#                 in_edge_indices = random.sample(range(len(inner_layers[i - 1])), in_edge_count)

#                 for edge_index in in_edge_indices:
#                     inner_layers[i][x][1].append(inner_layers[i - 1][edge_index][0]["id"])
#                     inner_layers[i - 1][edge_index][2].append(inner_layers[i][x][0]["id"])

#             if len(inner_layers[i][x][2]) == 0 and i < len(inner_layers) - 1:
#                 out_edge_count = 1
#                 if len(inner_layers[i + 1]) > 1:
#                     out_edge_count = random.randrange(1, len(inner_layers[i + 1]))
#                 out_edge_indices = random.sample(range(len(inner_layers[i + 1])), out_edge_count)

#                 for edge_index in out_edge_indices:
#                     inner_layers[i][x][2].append(inner_layers[i + 1][edge_index][0]["id"])
#                     inner_layers[i + 1][edge_index][1].append(inner_layers[i][x][0]["id"])
#             if i == len(inner_layers) - 1:
#                 output_layer[0][1].append(inner_layers[i][x][0]["id"])
#                 inner_layers[i][x][2].append(output_layer[0][0]["id"])
#     res = input_layer

#     for x in inner_layers:
#         res = res + x

#     res = res + output_layer

#     return res


# def get_task_perm(name):
#     global app_task_count
#     res = {
#         "name": name,
#         "mi": random.randrange(1, 10),
#         "ram": round(random.uniform(1, 4), 2),
#         "storage": round(random.uniform(1, 4), 2),
#         "data_in": random.randrange(1, 50),
#         "data_out": random.randrange(1, 50),
#         "cores": random.randrange(1, 4),
#         "offload": random.choices(
#             population=[0, 1],
#             weights=[0.3, 0.7],
#             k=1)[0],
#         "id": app_task_count
#     }
#     app_task_count = app_task_count + 1
#     return res


def log_info(applications, max_time, output_file_name):
    with open(output_file_name, 'w') as f:
        count = 0
        f.write(f"{max_time}\n")
        f.write(f"{len(applications)}\n")
        for tasks in applications:
            f.write(f"{applications[tasks]['num_layers']}\n")
            f.write(f"{applications[tasks]['offload_time']}\n")
            for task in applications[tasks]['layers']:
                f.write(
                    f'{task["name"]} {task["cloud"]} {task["edge"]} {task["mobile"]} {task["ram"]} {task["data_in"]} {task["data_out"]} {task["storage"]} {task["offload"]}\n')

                output_str = ""
                for index in task["dependencies"]:
                    output_str = f'{output_str}{index} '
                output_str = output_str.strip()
                f.write(f"{output_str}\n")
    return


def exp_sample(time, application_size):
    application_per_second = application_size / time
    uniform_val = random.uniform(0, sys.float_info.max)
    rand_num = uniform_val / sys.float_info.max - 1
    lambda_val = (1 * (math.exp(rand_num) / application_per_second))
    return lambda_val


def sorted_template(key, template):
    template["layers"].sort(key=lambda x: sum(x["dependencies"]) if len(x["dependencies"]) > 0 else -1)
    return template


def main(application_count, output_file_name, gpus):
    app_templates = generate_template(gpus)

    app_templates = {k: sorted_template(k, v) for k, v in app_templates.items()}

    applications = {x: generate_applications(x, app_templates, application_count) for x in app_templates.keys() + ["mixed"]}


    log_info(applications["inception_v4"], sim_time, output_file_name)
    return


def generate_applications(app_name, app_templates, application_count) -> dict:
    res = {}
    offload_time = 0
    for i in range(0, application_count):
        key_name = random.choice(list(app_templates.keys())) if app_name == "mixed" else app_name

        key = f"{i + 1}_{key_name}"
        res[key] = copy.deepcopy(app_templates[key_name])
        res[key]["offload_time"] = offload_time

        for x in range(0, len(res[key]["layers"])):
            res[key]["layers"][x]["offload"] = random.choices( population=[0, 1], weights=[0.3, 0.7], k=1)[0] if x > 1 and x < len(res[key]["layers"]) - 2 else 0
            res[key]["layers"][x]["name"] = f'{key}_{res[key]["layers"][x]["name"]}'
            res[key]["layers"][x]["ram"] = 0
            res[key]["layers"][x]["mobile"] = res[key]["layers"][x]["mobile"] / 1000
            res[key]["layers"][x]["edge"] = res[key]["layers"][x]["edge"] / 1000
            res[key]["layers"][x]["cloud"] = res[key]["layers"][x]["cloud"] / 1000
            res[key]["layers"][x]["data_in"] = res[key]["layers"][x]["data_in"] / 1000000
            res[key]["layers"][x]["data_out"] = res[key]["layers"][x]["data_out"] / 1000000
            res[key]["layers"][x]["storage"] = res[key]["layers"][x]["data_out"] + res[key]["layers"][x]["data_in"]
        offload_time = offload_time + exp_sample(sim_time, application_count)
    return res


def generate_template(gpus) -> dict:
    template_files = {_[:-len(json_ext)]:json.load(open(f"{templates_dir}/{_}", 'r')) for _ in os.listdir(templates_dir) if _.endswith(json_ext)}
    gpus_to_retrieve = set(gpus.values())
    performance_values = {_: {k.split(execution_file_break_char)[1][:-len(json_ext)]: json.load(open(f"{execution_dir}/{k}", 'r')) for k in os.listdir(execution_dir) if k.endswith(json_ext) and k.split(execution_file_break_char)[0] == _ } for _ in gpus_to_retrieve}

    for key in template_files.keys():
        for i in range(0, len(template_files[key]["layers"])):
            for ky in gpus.keys():
                template_files[key]["layers"][i][ky] = performance_values[gpus[ky]][key]["layers"][i]["runtime"]
    return template_files


if __name__ == "__main__":
    application_count = 1
    output_file_name = "applications_file.txt"
    gpus = {
        "mobile": "ADRENO_640", 
        "cloud":  "RTX_3060",
        "edge":    "RTX_3060" }
    if len(sys.argv) > 1:
        application_count = int(sys.argv[1])
        output_file_name = str(sys.argv[2])
        gpus["mobile"] = sys.argv[3]
        gpus["cloud"] = sys.argv[4]
        gpus["edge"] = sys.argv[5]
    main(application_count, output_file_name, gpus)
