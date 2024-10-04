from hdfs import InsecureClient

# Define the path to the dataset
hdfs_path = "http://hadoop-namenode:9820"  # Update with your HDFS namenode URL
dataset_path = "/project/as-skitter.txt"  # HDFS file path

# Create an HDFS client
client = InsecureClient(hdfs_path)

def parse_line(line):
    parts = line.split("\t")
    if len(parts) == 2:
        return (parts[0], parts[1])
    else:
        return None

# Load the dataset and parse lines
edges = []
with client.read(dataset_path) as file:
    for line in file:
        parsed_line = parse_line(line.strip())
        if parsed_line:
            edges.append(parsed_line)

# Create vertices set from edges
vertices = {src for src, dst in edges} | {dst for src, dst in edges}

# Basic graph operations
num_vertices = len(vertices)
num_edges = len(edges)
print(f"Number of vertices: {num_vertices}")
print(f"Number of edges: {num_edges}")

# Perform linear search for a specific user ID
target_user_id = "1691593"  # Update with the user ID you want to search for
user_found = target_user_id in vertices

# Collect the search result
result = f"User ID {target_user_id} {'found' if user_found else 'not found'} in the dataset."

# Write the result to a local file
output_path = "outputtest.txt"  # Specify the local output path
with open(output_path, 'w') as output_file:
    output_file.write(result + "\n")

# Find the neighbors of the specified user
neighbors = [edge for edge in edges if target_user_id in edge]
print(f"Neighbors of user {target_user_id}:")
for neighbor in neighbors:
    print(neighbor)
