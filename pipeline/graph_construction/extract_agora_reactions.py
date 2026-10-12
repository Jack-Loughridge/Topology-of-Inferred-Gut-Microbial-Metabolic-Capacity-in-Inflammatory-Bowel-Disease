import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd

# Folder containing all the AGORA *.xml models (what you just listed)
SBML_DIR = "/home/Jack/Real_Data/AGORA2_SBML_clean"

# Output file with all reactions across all models
OUT_PATH = "/home/Jack/Real_Data/AGORA_reactions.parquet"


def parse_sbml_file(path: str):
    """
    Parse a single SBML file and return a list of reaction dicts with:
      - model_id (SBML model id or filename stem)
      - species_file (XML filename)
      - reaction_id
      - inputs  (semicolon-separated metabolite ids)
      - outputs (semicolon-separated metabolite ids)
      - catalysts (here: the model_id, i.e. the microbe)
    """
    reactions = []

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        print(f"[!] Failed to parse {path}: {e}")
        return reactions

    # Detect namespace, e.g. {http://www.sbml.org/sbml/level3/version1/core}sbml
    if root.tag.startswith("{"):
        uri = root.tag[1:].split("}")[0]
        ns = {"sbml": uri}
        model = root.find("sbml:model", ns)
        reaction_xpath = ".//sbml:reaction"
        list_reactants = "sbml:listOfReactants/sbml:speciesReference"
        list_products = "sbml:listOfProducts/sbml:speciesReference"
    else:
        ns = {}
        model = root.find("model")
        reaction_xpath = ".//reaction"
        list_reactants = "listOfReactants/speciesReference"
        list_products = "listOfProducts/speciesReference"

    filename = os.path.basename(path)
    model_id = None
    if model is not None:
        model_id = model.get("id") or model.get("name")

    if not model_id:
        # fall back to filename without .xml
        model_id = os.path.splitext(filename)[0]

    for rxn in root.findall(reaction_xpath, ns):
        rxn_id = rxn.get("id") or rxn.get("name")
        if not rxn_id:
            # make *some* id so the row isn’t lost
            rxn_id = f"{model_id}_unnamed_{len(reactions)}"

        reactants = []
        products = []

        for sr in rxn.findall(list_reactants, ns):
            sp = sr.get("species")
            if sp:
                reactants.append(sp)

        for sr in rxn.findall(list_products, ns):
            sp = sr.get("species")
            if sp:
                products.append(sp)

        # de-duplicate but keep deterministic order
        reactants = sorted(set(reactants))
        products = sorted(set(products))

        reactions.append(
            {
                "model_id": model_id,
                "species_file": filename,
                "reaction_id": rxn_id,
                "inputs": ";".join(reactants),
                "outputs": ";".join(products),
                # treat the microbe (model) as the "catalyst"
                "catalysts": model_id,
            }
        )

    return reactions


def main():
    sbml_dir = os.path.abspath(SBML_DIR)
    if not os.path.isdir(sbml_dir):
        raise SystemExit(f"SBML directory not found: {sbml_dir}")

    xml_paths = sorted(glob.glob(os.path.join(sbml_dir, "*.xml")))
    if not xml_paths:
        raise SystemExit(f"No .xml files found under {sbml_dir}")

    all_rows = []
    print(f"[*] Found {len(xml_paths)} SBML models, extracting reactions…")

    for path in tqdm(xml_paths, desc="Parsing SBML models"):
        rows = parse_sbml_file(path)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("[!] No reactions extracted – something went wrong.")

    df = pd.DataFrame(all_rows)
    print(f"[*] Extracted {len(df)} reactions total.")

    out_path = os.path.abspath(OUT_PATH)
    df.to_parquet(out_path)
    print(f"[*] Saved reaction table to: {out_path}")


if __name__ == "__main__":
    main()
