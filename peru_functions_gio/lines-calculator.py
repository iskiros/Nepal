import os

directory = 'chemweathering/functions_gio'
total_lines = 0

for filename in os.listdir(directory):
    if filename.endswith('.py'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            total_lines += len(lines)

print(f'Total number of lines in .py files: {total_lines}')
