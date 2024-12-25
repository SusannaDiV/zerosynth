from rdkit import Chem

# Input and output file paths
input_file = "protein.pdb"
output_file = "protein.sdf"

# Load the PCB file
mol = Chem.MolFromPDBFile(input_file)

# Check if the molecule was successfully loaded
if mol:
    # Write to SDF format
    w = Chem.SDWriter(output_file)
    w.write(mol)
    w.close()
    print("Conversion successful!")
else:
    print("Failed to load the PCB file.")
