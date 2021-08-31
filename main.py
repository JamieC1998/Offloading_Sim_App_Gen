import random
import math
import sys


def get_task_perm(name):
    return {
        "name": name,
        "mi": random.randrange(1, 10),
        "ram": round(random.uniform(1, 4), 2),
        "storage": round(random.uniform(1, 4), 2),
        "data_in": random.randrange(1, 50),
        "data_out": random.randrange(1, 50),
        "cores": random.randrange(1, 4),
        "offload": random.randrange(0, 1)
    }


def log_info(applications, max_time):
    count = 0
    print(max_time)
    print(len(applications))
    for tasks in applications:
        print(len(tasks))
        count = count + exp_sample(max_time, len(applications))
        print(count)
        for task in tasks:
            item = task[0]
            print(f'{item["name"]} {item["mi"]} {item["ram"]} {item["data_in"]} {item["data_out"]} {item["storage"]} {item["offload"]} {item["cores"]}')

            output_str = ""
            for index in task[1]:
                output_str = f'{output_str}{index} '
            output_str = output_str.strip()
            print(output_str)
    return


def exp_sample(time, application_size):
    # // Introduce a delay for the next node
    # float randNum = ((float)rand()) / RAND_MAX - 1;
    # int this_lambda = 1000;
    # if (randNum > -1)
    #     this_lambda = -(1000 * (float)(log1pf(randNum) / ((float)rate)));
    # std::cout << "Rand num " << randNum << " Lambda is " << this_lambda << std::endl;
    # std::this_thread::sleep_for(std::chrono::milliseconds(this_lambda));

    application_per_second = application_size / time
    uniform_val = random.uniform(0, sys.float_info.max)
    rand_num = uniform_val / sys.float_info.max - 1
    lambda_val = (1 * (math.exp(rand_num) / application_per_second))
    return lambda_val


def main(application_count):
    max_layer_size = 5
    max_time = random.uniform(15, 20)
    applications = []

    for x in range(0, application_count):
        applications.append(
            generate_applications(max_layer_size, x))

    log_info(applications, max_time)
    return


def get_last_layer_size(inner_layers):
    if len(inner_layers) != 0:
        return len(inner_layers[len(inner_layers) - 1])
    return 0


def retrieve_options(layer_list, last_layer_size, input_layer_size):
    number_links = 1
    layer_list_size = len(layer_list)

    if last_layer_size > 1:
        number_links = random.randrange(1, last_layer_size)

    options = random.sample(range(last_layer_size), number_links)
    total_index_size = input_layer_size
    for j in range(0, layer_list_size - 1):
        total_index_size = total_index_size + len(layer_list[j])

    options = list(map(lambda option: option + total_index_size, options))
    return options


def generate_applications(max_layer_size, count):
    input_layer_size = random.randrange(1, max_layer_size)
    output_layer_size = random.randrange(1, max_layer_size)
    inner_layer_count = random.randrange(1, max_layer_size)

    input_layer = []
    inner_layers = []
    output_layer = []
    res = []

    # GENERATING INPUT LAYER
    for i in range(input_layer_size):
        input_layer.append((get_task_perm(f'APP_{count}_Input_Layer_{i}'), []))

    # GENERATING INNER LAYERS
    for i in range(inner_layer_count):
        layer = []
        layer_size = random.randrange(1, max_layer_size)

        for x in range(layer_size):
            task = get_task_perm(f'APP_{count}_Inner_Layer_{i}:{x}')

            last_layer_size = get_last_layer_size(inner_layers)

            options = []

            if last_layer_size != 0:
                options = retrieve_options(inner_layers, last_layer_size, input_layer_size)
            else:
                number_links = 0
                if input_layer_size > 1:
                    number_links = random.randrange(1, input_layer_size)

                options = random.sample(range(input_layer_size), number_links)

            layer.append((task, options))
        inner_layers.append(layer)

    # GENERATING OUTPUT LAYER
    for i in range(0, output_layer_size):
        task = get_task_perm(f'APP_{count}_Output_Layer_{i}')

        last_layer_size = get_last_layer_size(inner_layers)

        options = retrieve_options(inner_layers, last_layer_size, input_layer_size)

        output_layer.append((task, options))

    res = input_layer

    for x in inner_layers:
        res = res + x

    res = res + output_layer

    return res


if __name__ == "__main__":
    application_count = 1
    if len(sys.argv) > 1:
        application_count = int(sys.argv[1])
    main(application_count)
