import json
import re

def update_notebook(filepath):
    with open(filepath, "r") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell["cell_type"] != "code": continue

        source = cell["source"]

        new_source = []
        for i, line in enumerate(source):
            # 1. Update config defaults

            # Match directly python assignments (e.g. STAGE1_EPOCHS = 10)
            if re.search(r'^STAGE1_EPOCHS\s*=\s*\d+', line):
                line = re.sub(r'(STAGE1_EPOCHS\s*=\s*)\d+', r'\g<1>20', line)
            elif re.search(r'^STAGE2_EPOCHS\s*=\s*\d+', line):
                line = re.sub(r'(STAGE2_EPOCHS\s*=\s*)\d+', r'\g<1>40', line)

            # Match strings written to files (e.g. "STAGE1_EPOCHS = 10\n",)
            elif re.search(r'STAGE1_EPOCHS\s*=\s*\d+\\n', line):
                line = re.sub(r'(STAGE1_EPOCHS\s*=\s*)\d+(\\n)', r'\g<1>20\g<2>', line)
            elif re.search(r'STAGE2_EPOCHS\s*=\s*\d+\\n', line):
                line = re.sub(r'(STAGE2_EPOCHS\s*=\s*)\d+(\\n)', r'\g<1>40\g<2>', line)

            # Clean up hardcoded run epochs
            if 'stage1_epochs=1, stage2_epochs=1' in line:
                line = line.replace(', stage1_epochs=1, stage2_epochs=1', '')
                line = line.replace('stage1_epochs=1, stage2_epochs=1', '')
            if 'stage1_epochs=2, stage2_epochs=2' in line:
                line = line.replace(', stage1_epochs=2, stage2_epochs=2', '')
                line = line.replace('stage1_epochs=2, stage2_epochs=2', '')

            new_source.append(line)

        cell["source"] = new_source

    with open(filepath, "w") as f:
        json.dump(nb, f, indent=2)

update_notebook("Brain_Tumor_Classfication_ViT_Final.ipynb")

# Update string block config generator
with open("Brain_Tumor_Classfication_ViT_Final.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] != "code": continue

    # Check if this cell writes config_content_to_add
    is_config_cell = any("config_content_to_add =" in line for line in cell["source"])
    if is_config_cell:
        # We should append NUM_LAYERS and EMBED_DIM to config_content_to_add if they don't exist
        new_source = []
        for line in cell["source"]:
            new_source.append(line)
            if "STAGE2_EPOCHS = 40\\n\"," in line: # Remember we changed this
                new_source.append("        \"NUM_LAYERS = 2\\n\",\n")
                new_source.append("        \"EMBED_DIM = 256\\n\",\n")
            elif "STAGE2_EPOCHS = 40\n" in line:
                new_source.append("NUM_LAYERS = 2\n")
                new_source.append("EMBED_DIM = 256\n")

        final_source = []
        for line in new_source:
            if "[\"DEVICE\", \"BATCH_SIZE\", \"K_FOLDS\", \"RESULTS_DIR\", \"CHECKPOINT_DIR\", \"STAGE1_EPOCHS\", \"STAGE2_EPOCHS\"]" in line:
                line = line.replace("STAGE2_EPOCHS\"]", "STAGE2_EPOCHS\", \"NUM_LAYERS\", \"EMBED_DIM\"]")
            final_source.append(line)

        cell["source"] = final_source

with open("Brain_Tumor_Classfication_ViT_Final.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Patching done!")
