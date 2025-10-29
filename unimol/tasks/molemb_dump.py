# save_mol_embeddings.py
import os, numpy as np, torch, unicore, argparse
from tqdm import tqdm

from unicore.tasks import UnicoreTask, register_task
from unimol.data import (AffinityDataset, CroppingPocketDataset,
                         CrossDistanceDataset, DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset, NormalizeDockingPoseDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord,
                         RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset, AffinityMolDataset, AffinityPocketDataset, ResamplingDataset, LazySDFDataset)

@register_task("extract_mol_embeddings")
class MolEmbDumper(UnicoreTask):
    def __init__(self, model, device="cuda"):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def dump(self, mol_data_path, save_dir, bsz=64):
        os.makedirs(save_dir, exist_ok=True)
        # data_path = f"./data/DUD-E/raw/all/{target}/mols.lmdb"
        target = mol_data_path.split("/")[-1].split(".lmdb")[0]
        mol_dataset = self.load_mols_dataset(mol_data_path, "atoms", "coordinates")

        loader = torch.utils.data.DataLoader(
            mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater
        )

        reps, names, labels = [], [], []
        for sample in tqdm(loader, desc=f"Encode mols({target})"):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et   = sample["net_input"]["mol_src_edge_type"]
            st   = sample["net_input"]["mol_src_tokens"]

            pad = st.eq(self.model.mol_model.padding_idx)
            x   = self.model.mol_model.embed_tokens(st)
            n   = dist.size(-1)
            gbf = self.model.mol_model.gbf(dist, et)
            gab = self.model.mol_model.gbf_proj(gbf).permute(0,3,1,2).contiguous().view(-1, n, n)

            out = self.model.mol_model.encoder(x, padding_mask=pad, attn_mask=gab)
            rep = out[0][:,0,:]                                  # [B, D]
            rep = self.model.mol_project(rep)                    # [B, d_proj]
            rep = rep / rep.norm(dim=-1, keepdim=True)           # cosine/IP 可互换
            reps.append(rep.detach().cpu().to(torch.float32).numpy())

            names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())

        reps   = np.concatenate(reps, axis=0).astype(np.float32) # [N, d]
        labels = np.asarray(labels, dtype=np.int32)
        # 名称用.npy会是object数组，建议写成txt，或用np.savez
        np.save(os.path.join(save_dir, f"{target}_mol_reps.npy"),   reps)
        np.save(os.path.join(save_dir, f"{target}_mol_labels.npy"), labels)
        with open(os.path.join(save_dir, f"{target}_mol_names.txt"), "w") as f:
            for s in names: f.write(s+"\n")

        print("Saved:", reps.shape, labels.shape, "to", save_dir)

    
    def load_mols_dataset(self, data_path, atoms, coords, **kwargs):
        # data_path = '/vepfs-mlp2/mlp-public/shikunfeng/Project/DrugCLIP/data/inami/Enamine_screening_collection_202508.sdf'
        if data_path.endswith(".sdf"):
            dataset = LazySDFDataset(data_path)
        else:
            dataset = LMDBDataset(data_path)
        label_dataset = KeyDataset(dataset, "label")
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        
        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "target":  RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="e.g., 'ampc'")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save embeddings")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # 假设 model 已经定义并加载了预训练权重
    model = ...  # Load your pretrained model here

    dumper = MolEmbDumper(model, device="cuda")
    dumper.dump(args.target, args.save_dir, bsz=args.batch_size)