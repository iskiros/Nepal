import pandas as pd
import pdfplumber



########################### FUNCTION DEFINITION ###########################
# - Reads pdf produced by PhreeQC, extracts data and places it into a dataframe.
# - As of yet not called by the Master script, ran initially as an experiment



##################################################################

# Open the PDF file
file_path = '/Users/enrico/Downloads/batch_analysis_for_Si_calculation.pdf'

# Initialize a list to store the combined data for all solutions
all_solutions_data = []

##################################################################


##################################################################

####### PDF OPENING #######

# Use pdfplumber to extract text from the PDF
with pdfplumber.open(file_path) as pdf:
    # Extract text from all pages starting from page 3
    text_plumber = ''
    for page in pdf.pages[2:]:
        text_plumber += page.extract_text() if page.extract_text() else ''

# Manually parse the text to extract data for each solution
solution_start_marker = "Initial solution "
solution_end_marker = "Initial solution "

##################################################################



##################################################################

####### DATA EXTRACTION #######

# Loop through each solution in the text
for solution_number in range(1, 53):  # Assuming there are 52 solutions
    try:
        # Define the markers for the current and next solution
        current_solution_marker = f"{solution_start_marker}{solution_number}"
        next_solution_marker = f"{solution_end_marker}{solution_number + 1}"

        # Extract the section for the current solution
        solution_start_index = text_plumber.find(current_solution_marker)
        solution_end_index = text_plumber.find(next_solution_marker, solution_start_index)
        
        # Handle the last solution, which doesn't have a subsequent marker
        if solution_end_index == -1:
            solution_end_index = len(text_plumber)

        solution_text = text_plumber[solution_start_index:solution_end_index]

        # Extract the "Distribution of Species" section
        species_section_start = solution_text.find("Distribution of species")
        species_section_end = solution_text.find("Saturation indices")
        species_section = solution_text[species_section_start:species_section_end].strip()

        # Extract the "Saturation Indices" section
        saturation_indices_section_start = solution_text.find("Saturation indices")
        saturation_indices_section = solution_text[saturation_indices_section_start:].strip()

        # Parse the "Distribution of Species" section
        species_lines = species_section.split("\n")
        species_data = []
        for line in species_lines:
            parts = line.split()
            if len(parts) >= 6 and not line.startswith(("Species", "-")):
                species_data.append({
                    "Species": parts[0],
                    "Molality": parts[1],
                    "Activity": parts[2],
                    "Log Activity": parts[4]
                })

        # Parse the "Saturation Indices" section
        saturation_lines = saturation_indices_section.split("\n")
        saturation_indices_data = []
        for line in saturation_lines:
            parts = line.split()
            if len(parts) >= 5 and not line.startswith(("Phase", "-")):
                saturation_indices_data.append({
                    "Phase": parts[0],
                    "SI**": parts[1]
                })

        # Combine all data into a single row
        combined_data = {
            "Solution Number": solution_number,
            **{f"{species['Species']} Activity": species["Activity"] for species in species_data},
            **{f"{species['Species']} Log Activity": species["Log Activity"] for species in species_data},
            **{f"{phase['Phase']} SI**": phase["SI**"] for phase in saturation_indices_data}
        }

        # Append the data for this solution to the list
        all_solutions_data.append(combined_data)
    
    except Exception as e:
        print(f"An error occurred while processing solution {solution_number}: {e}")

##################################################################



##################################################################

# Create a DataFrame with the combined data for all solutions
combined_df = pd.DataFrame(all_solutions_data)

# Save the combined data to a CSV file
combined_csv_path = '/Users/enrico/Downloads/all_solutions_combined_analysis_new.csv'
combined_df.to_csv(combined_csv_path, index=False)

print(f"Data for all solutions has been saved to {combined_csv_path}")
