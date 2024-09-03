

instance_dir = "Instances/"

# Optimal objective values for all instances
opt_obj = {}
with open(f"{instance_dir}obj.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        inst, obj = line[:-1].split(" ")
        opt_obj[inst] = obj


# Splitting these into a separate file for each instance
for i in range(1, 13):
    filename = f"{instance_dir}gap{i}.txt"
    with open(filename, "r") as f:
        lines = f.readlines()

    num_instances = int(lines[0][:-1])
    startline = lines[1]
    dividers = [l for l in range(len(lines)) if lines[l] == startline]

    assert num_instances == len(dividers)
    dividers.append(len(lines))

    # Printing individual instance data
    for instance_num in range(1, num_instances+1):

        params = ''.join(lines[dividers[instance_num - 1] : dividers[instance_num]]).replace("\n", "")
        params = params.split(" ")[1:]

        machines, jobs = map(int, params[:2])
        assert len(params) == 2 + 2*machines*jobs + machines

        instance_name = f"c{machines}{jobs}-{instance_num}"
        newfile = instance_dir + instance_name + ".txt"

        new_lines = []
        new_lines.append(" ".join(params[:2]) + " " + opt_obj[instance_name] + "\n")
        id = 2

        for m in range(2*machines):
            new_lines.append(" ".join(params[id: id+jobs]) + "\n")
            id += jobs
        new_lines.append(" ".join(params[id: ]) + "\n")

        with open(newfile, "w") as f:
            f.writelines(new_lines)