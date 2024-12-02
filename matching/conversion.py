import pandas as pd

# Initialize list for storing edges
edges = []

# Path to your .mtx file
input_path = "C:\\Users\\ps232\Downloads\web-uk-2002-all\web-uk-2002-all.mtx"

# Open the .mtx file
with open(input_path, "r") as mtx_file:
    for line in mtx_file:
        if line.startswith('%'):
            continue  # Skip comments
        values = line.strip().split()
        if len(values) == 3:  # Skip the header line with dimensions
            continue
        vertex1, vertex2 = int(values[0]), int(values[1])
        if vertex1 < vertex2:  # Avoid duplicate edges in undirected graphs
            edges.append((vertex1, vertex2))

# Convert to DataFrame
edges_df = pd.DataFrame(edges, columns=["Vertex1", "Vertex2"])

# Save to CSV locally
output_csv_path = "C:\\Users\ps232\Downloads\web-uk-2002-all\web-uk-2002.csv"
edges_df.to_csv(output_csv_path, index=False)
print(f"CSV saved locally at {output_csv_path}")
