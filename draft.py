from datasets.MolGraph_Construction import smiles_to_Molgraph

smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
a = smiles_to_Molgraph(smiles)
print(a)