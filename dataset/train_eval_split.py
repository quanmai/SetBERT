import jsonlines, json

L = 50000
R = 0.8
M = int(L * R)

def read_jsonl(file_path):
    res = []
    with open(file_path, 'r') as file:
        for (i, line) in enumerate(file):
            try:
                # Parse the JSON object and append to the list
                data = json.loads(line)
                res.append(data)
            except json.JSONDecodeError as e:
                print(f"Error at line {i + 1} in file {file_path}")
                print(f"Error decoding JSON: {e}")
    return res

def write_jsonl(file_path, data):
    with open(file_path, 'w') as file:
        for obj in data:
            file.write(json.dumps(obj))
            file.write('\n')

def main():
    or_data = read_jsonl('boolean/or.jsonl')
    and_data = read_jsonl('boolean/and.jsonl')
    not_data = read_jsonl('boolean/not.jsonl')
    train_data = or_data[:M] + and_data[:M] + not_data[:M]
    eval_data = or_data[M:] + and_data[M:] + not_data[M:]

    write_jsonl('train.jsonl', train_data)
    write_jsonl('eval.jsonl', eval_data)

if __name__ == '__main__':
    main()
