#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import numpy as np
from tqdm import tqdm
import unicore

# from unimol.tasks.molemb_dump import MolEmbDumper

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve



def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)


    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(args)


    model.eval()
    
    if args.test_task == "extract_feats":
        # dumper = MolEmbDumper(model, device="cuda" if use_cuda else "cpu")
        # dumper.dump(args.mol_data_path, args.save_dir, bsz=args.batch_size)
        task.encode_mols_once(model, args.mol_data_path, args.save_dir,  "atoms", "coordinates")
    
    if args.test_task == "retrival":
        mol_names, topk = task.retrieve_mols(model, args.mol_data_path, args.pock_path, args.molemb_path, k=1000)

        # write results to file
        with open(os.path.join(args.save_dir, "retrival_results.txt"), "w") as f:
            for mol_name, score in zip(mol_names, topk):
                f.write(f"{mol_name}\t" + f"{score}" + "\n")
        
    if args.test_task=="DUDE":
        task.test_dude(model)

    elif args.test_task=="PCBA":
        task.test_pcba(model)


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--test-task", type=str, default="DUDE", help="test task", choices=["DUDE", "PCBA", "extract_feats", "retrival"])
    parser.add_argument("--mol_data_path", type=str, default="/vepfs-mlp2/mlp-public/shikunfeng/Project/DrugCLIP/data/inami/Enamine_202508_3d.lmdb", help="")
    parser.add_argument("--save_dir", type=str, default="/vepfs-mlp2/mlp-public/shikunfeng/Project/DrugCLIP/data/inami/mol_feats", help="test task")
    parser.add_argument("--molemb_path", type=str, default="/vepfs-mlp2/mlp-public/shikunfeng/Project/DrugCLIP/data/inami/mol_feats/Enamine_202508_3d.lmdb.pkl", help="retrival task for mol embeddings")
    parser.add_argument("--pock_path", type=str, default="/vepfs-mlp2/mlp-public/shikunfeng/Project/DrugCLIP/data/inami/pocket_data.lmdb", help="retrival task for mol embeddings")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
