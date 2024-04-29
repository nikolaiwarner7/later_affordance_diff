import json

# Load the JSON data
json_data = json.load(open('train_captions_frame_mapping_gpu_v2.json'))

def analyze_prefixes(json_data):
    different_prefix_count = 0
    prefix_usage = {}  # Tracks the number of times each prefix is used

    for key, values in json_data.items():
        prefixes = set(value.split('_frame')[0] for value in values)
        if len(prefixes) > 1:
            different_prefix_count += 1
        else:
            # For videos sharing the same prefix, update their count in prefix_usage
            prefix = prefixes.pop()  # Since all prefixes are the same, take the only element
            if prefix in prefix_usage:
                prefix_usage[prefix] += 1
            else:
                prefix_usage[prefix] = 1

    # Count how many prefixes are used in more than one key
    duplicate_prefix_count = sum(1 for count in prefix_usage.values() if count > 1)

    return different_prefix_count, duplicate_prefix_count

# Perform the analysis
different_prefix_count, duplicate_prefix_count = analyze_prefixes(json_data)

print(f'Number of keys with values that don\'t share the same prefix: {different_prefix_count}')
print(f'Number of unique prefixes used in multiple keys: {duplicate_prefix_count}')
