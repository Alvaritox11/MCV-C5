from cleanfid import fid

real_dir = "/ghome/group05/datasets/VizWiz/train"
synth_dir = "/ghome/group05/datasets/synthetic_vizwiz_blurry/images"
deartdir = "/ghome/group05/datasets/deART/images"

print("Calculating FID...")
score = fid.compute_fid(real_dir, deartdir)
print(f"✅ FID Score vs deART: {score:.3f}")