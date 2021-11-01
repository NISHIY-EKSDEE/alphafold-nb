
import os
import subprocess

import jax
if jax.local_devices()[0].platform == 'tpu':
  raise RuntimeError('Colab TPU runtime not supported. Change it to GPU via Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')
elif jax.local_devices()[0].platform == 'cpu':
  raise RuntimeError('Colab CPU runtime not supported. Change it to GPU via Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')

print('Making a prediction')

# Enter the amino acid sequence to fold
sequence = 'MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEKGEHEQAAHHADTAYAHHKHAEEHAAQAAKHDAEHHAPKPH'  #@param {type:"string"}

MIN_SEQUENCE_LENGTH = 16
MAX_SEQUENCE_LENGTH = 2500

# Remove all whitespaces, tabs and end lines; upper-case
sequence = sequence.translate(str.maketrans('', '', ' \n\t')).upper()
aatypes = set('ACDEFGHIKLMNPQRSTVWY')  # 20 standard aatypes
if not set(sequence).issubset(aatypes):
  raise Exception(f'Input sequence contains non-amino acid letters: {set(sequence) - aatypes}. AlphaFold only supports 20 standard amino acids as inputs.')
if len(sequence) < MIN_SEQUENCE_LENGTH:
  raise Exception(f'Input sequence is too short: {len(sequence)} amino acids, while the minimum is {MIN_SEQUENCE_LENGTH}')
if len(sequence) > MAX_SEQUENCE_LENGTH:
  raise Exception(f'Input sequence is too long: {len(sequence)} amino acids, while the maximum is {MAX_SEQUENCE_LENGTH}. Please use the full AlphaFold system for long sequences.')


# Search against genetic databases
import sys
sys.path.append('/workspace/miniconda/lib/python3.9/site-packages')

import os
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'

from urllib import request
from concurrent import futures
import json
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import py3Dmol

from alphafold.model import model
from alphafold.model import config
from alphafold.model import data

from alphafold.data import parsers
from alphafold.data import pipeline
from alphafold.data.tools import jackhmmer

from alphafold.common import protein

from alphafold.relax import relax
from alphafold.relax import utils

# Color bands for visualizing plddt
PLDDT_BANDS = [(0, 50, '#FF7D45'),
               (50, 70, '#FFDB13'),
               (70, 90, '#65CBF3'),
               (90, 100, '#0053D6')]

# --- Find the closest source ---
test_url_pattern = 'https://storage.googleapis.com/alphafold-colab{:s}/latest/uniref90_2021_03.fasta.1'
ex = futures.ThreadPoolExecutor(3)
def fetch(source):
  request.urlretrieve(test_url_pattern.format(source))
  return source
fs = [ex.submit(fetch, source) for source in ['', '-europe', '-asia']]
source = None
for f in futures.as_completed(fs):
  source = f.result()
  ex.shutdown()
  break

# --- Search against genetic databases ---
with open('target.fasta', 'wt') as f:
  f.write(f'>query\n{sequence}')

# Run the search against chunks of genetic databases

jackhmmer_binary_path = '/usr/bin/jackhmmer'
dbs = []

num_jackhmmer_chunks = {'uniref90': 59, 'smallbfd': 17, 'mgnify': 71}
total_jackhmmer_chunks = sum(num_jackhmmer_chunks.values())

print('Searching uniref90')
jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
    binary_path=jackhmmer_binary_path,
    database_path=f'https://storage.googleapis.com/alphafold-colab{source}/latest/uniref90_2021_03.fasta',
    get_tblout=True,
    num_streamed_chunks=num_jackhmmer_chunks['uniref90'],
    z_value=135301051)
dbs.append(('uniref90', jackhmmer_uniref90_runner.query('target.fasta')))

print('Searching smallbfd')
jackhmmer_smallbfd_runner = jackhmmer.Jackhmmer(
    binary_path=jackhmmer_binary_path,
    database_path=f'https://storage.googleapis.com/alphafold-colab{source}/latest/bfd-first_non_consensus_sequences.fasta',
    get_tblout=True,
    num_streamed_chunks=num_jackhmmer_chunks['smallbfd'],
    z_value=65984053)
dbs.append(('smallbfd', jackhmmer_smallbfd_runner.query('target.fasta')))

print('Searching mgnify')
jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
    binary_path=jackhmmer_binary_path,
    database_path=f'https://storage.googleapis.com/alphafold-colab{source}/latest/mgy_clusters_2019_05.fasta',
    get_tblout=True,
    num_streamed_chunks=num_jackhmmer_chunks['mgnify'],
    z_value=304820129)
dbs.append(('mgnify', jackhmmer_mgnify_runner.query('target.fasta')))


# --- Extract the MSAs and visualize ---
# Extract the MSAs from the Stockholm files.
# NB: deduplication happens later in pipeline.make_msa_features.

mgnify_max_hits = 501

msas = []
deletion_matrices = []
full_msa = []
for db_name, db_results in dbs:
    unsorted_results = []
    for i, result in enumerate(db_results):
        msa, deletion_matrix, target_names = parsers.parse_stockholm(result['sto'])
        e_values_dict = parsers.parse_e_values_from_tblout(result['tbl'])
        e_values = [e_values_dict[t.split('/')[0]] for t in target_names]
        zipped_results = zip(msa, deletion_matrix, target_names, e_values)
        if i != 0:
           # Only take query from the first chunk
           zipped_results = [x for x in zipped_results if x[2] != 'query']
        unsorted_results.extend(zipped_results)
    sorted_by_evalue = sorted(unsorted_results, key=lambda x: x[3])
    db_msas, db_deletion_matrices, _, _ = zip(*sorted_by_evalue)
    if db_msas:
        if db_name == 'mgnify':
            db_msas = db_msas[:mgnify_max_hits]
            db_deletion_matrices = db_deletion_matrices[:mgnify_max_hits]
        full_msa.extend(db_msas)
        msas.append(db_msas)
        deletion_matrices.append(db_deletion_matrices)
        msa_size = len(set(db_msas))
        print(f'{msa_size} Sequences Found in {db_name}')

deduped_full_msa = list(dict.fromkeys(full_msa))
total_msa_size = len(deduped_full_msa)
print(f'\n{total_msa_size} Sequences Found in Total\n')

aa_map = {restype: i for i, restype in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ-')}
msa_arr = np.array([[aa_map[aa] for aa in seq] for seq in deduped_full_msa])
num_alignments, num_res = msa_arr.shape

fig = plt.figure(figsize=(12, 3))
plt.title('Per-Residue Count of Non-Gap Amino Acids in the MSA')
plt.plot(np.sum(msa_arr != aa_map['-'], axis=0), color='black')
plt.ylabel('Non-Gap Count')
plt.yticks(range(0, num_alignments + 1, max(1, int(num_alignments / 3))))
plt.savefig('search_result.png')


# Run AlphaFold

# --- Run the model ---
model_names = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_2_ptm']

def _placeholder_template_feats(num_templates_, num_res_):
  return {
      'template_aatype': np.zeros([num_templates_, num_res_, 22], np.float32),
      'template_all_atom_masks': np.zeros([num_templates_, num_res_, 37, 3], np.float32),
      'template_all_atom_positions': np.zeros([num_templates_, num_res_, 37], np.float32),
      'template_domain_names': np.zeros([num_templates_], np.float32),
      'template_sum_probs': np.zeros([num_templates_], np.float32),
  }

output_dir = '/workspace/prediction'
os.makedirs(output_dir, exist_ok=True)

plddts = {}
pae_outputs = {}
unrelaxed_proteins = {}

for model_name in model_names:
    print(f'Running {model_name}')
    num_templates = 0
    num_res = len(sequence)

    feature_dict = {}
    feature_dict.update(pipeline.make_sequence_features(sequence, 'test', num_res))
    feature_dict.update(pipeline.make_msa_features(msas, deletion_matrices=deletion_matrices))
    feature_dict.update(_placeholder_template_feats(num_templates, num_res))

    cfg = config.model_config(model_name)
    params = data.get_model_haiku_params(model_name, '/workspace/alphafold/data')
    model_runner = model.RunModel(cfg, params)
    processed_feature_dict = model_runner.process_features(feature_dict,
                                                           random_seed=0)
    prediction_result = model_runner.predict(processed_feature_dict)

    mean_plddt = prediction_result['plddt'].mean()

    if 'predicted_aligned_error' in prediction_result:
        pae_outputs[model_name] = (
            prediction_result['predicted_aligned_error'],
            prediction_result['max_predicted_aligned_error']
        )
    else:
        # Get the pLDDT confidence metrics. Do not put pTM models here as they
        # should never get selected.
        plddts[model_name] = prediction_result['plddt']

    # Set the b-factors to the per-residue plddt.
    final_atom_mask = prediction_result['structure_module']['final_atom_mask']
    b_factors = prediction_result['plddt'][:, None] * final_atom_mask
    unrelaxed_protein = protein.from_prediction(processed_feature_dict,
                                                prediction_result,
                                                b_factors=b_factors)
    unrelaxed_proteins[model_name] = unrelaxed_protein

    # Delete unused outputs to save memory.
    del model_runner
    del params
    del prediction_result

# --- AMBER relax the best model ---
print(f'AMBER relaxation')
amber_relaxer = relax.AmberRelaxation(
    max_iterations=0,
    tolerance=2.39,
    stiffness=10.0,
    exclude_residues=[],
    max_outer_iterations=20)
# Find the best model according to the mean pLDDT.
best_model_name = max(plddts.keys(), key=lambda x: plddts[x].mean())
relaxed_pdb, _, _ = amber_relaxer.process(
    prot=unrelaxed_proteins[best_model_name])
# Finished AMBER relax.

# Construct multiclass b-factors to indicate confidence bands
# 0=very low, 1=low, 2=confident, 3=very high
banded_b_factors = []
for plddt in plddts[best_model_name]:
    for idx, (min_val, max_val, _) in enumerate(PLDDT_BANDS):
        if plddt >= min_val and plddt <= max_val:
            banded_b_factors.append(idx)
            break
banded_b_factors = np.array(banded_b_factors)[:, None] * final_atom_mask
to_visualize_pdb = utils.overwrite_b_factors(relaxed_pdb, banded_b_factors)


# Write out the prediction
pred_output_path = os.path.join(output_dir, 'selected_prediction.pdb')
with open(pred_output_path, 'w') as f:
  f.write(relaxed_pdb)


# --- Visualise the prediction & confidence ---
show_sidechains = True
def plot_plddt_legend():
    """Plots the legend for pLDDT."""
    thresh = [
                'Very low (pLDDT < 50)',
                'Low (70 > pLDDT > 50)',
                'Confident (90 > pLDDT > 70)',
                'Very high (pLDDT > 90)']

    colors = [x[2] for x in PLDDT_BANDS]

    plt.figure(figsize=(2, 2))
    for c in colors:
        plt.bar(0, 0, color=c)
    plt.legend(thresh, frameon=False, loc='center', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.title('Model Confidence', fontsize=20, pad=20)
    return plt

# Color the structure by per-residue pLDDT
color_map = {i: bands[2] for i, bands in enumerate(PLDDT_BANDS)}
view = py3Dmol.view(width=800, height=600)
view.addModelsAsFrames(to_visualize_pdb)
style = {'cartoon': {
            'colorscheme': {
            'prop': 'b',
            'map': color_map}
        }}
if show_sidechains:
    style['stick'] = {}
view.setStyle({'model': -1}, style)
view.zoomTo()

# grid = GridspecLayout(1, 2)
# out = Output()
# with out:
#     view.show()
# grid[0, 0] = out

# out = Output()
# with out:
#     plot_plddt_legend().show()
# grid[0, 1] = out

# Display pLDDT and predicted aligned error (if output by the model).
if pae_outputs:
  num_plots = 2
else:
  num_plots = 1

plt.figure(figsize=[8 * num_plots, 6])
plt.subplot(1, num_plots, 1)
plt.plot(plddts[best_model_name])
plt.title('Predicted LDDT')
plt.xlabel('Residue')
plt.ylabel('pLDDT')

if num_plots == 2:
    plt.subplot(1, 2, 2)
    pae, max_pae = list(pae_outputs.values())[0]
    plt.imshow(pae, vmin=0., vmax=max_pae, cmap='Greens_r')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Predicted Aligned Error')
    plt.xlabel('Scored residue')
    plt.ylabel('Aligned residue')

# Save pLDDT and predicted aligned error (if it exists)
pae_output_path = os.path.join(output_dir, 'predicted_aligned_error.json')
if pae_outputs:
    # Save predicted aligned error in the same format as the AF EMBL DB
    rounded_errors = np.round(pae.astype(np.float64), decimals=1)
    indices = np.indices((len(rounded_errors), len(rounded_errors))) + 1
    indices_1 = indices[0].flatten().tolist()
    indices_2 = indices[1].flatten().tolist()
    pae_data = json.dumps([{
        'residue1': indices_1,
        'residue2': indices_2,
        'distance': rounded_errors.flatten().tolist(),
        'max_predicted_aligned_error': max_pae.item()
    }],
                        indent=None,
                        separators=(',', ':'))
    with open(pae_output_path, 'w') as f:
        f.write(pae_data)