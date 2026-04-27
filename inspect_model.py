import h5py

f = h5py.File(r'backend/brain_tumor_mobilenetv2_fulltune.h5', 'r')

print("=== CLASSIFIER LAYERS ===")
for k in f['weights']['classifier'].keys():
    grp = f['weights']['classifier'][k]
    for name in grp.keys():
        print(f"  classifier[{k}][{name}] shape:", grp[name].shape)

print("\n=== FEATURES LAYERS ===")
for k in f['weights']['features'].keys():
    grp = f['weights']['features'][k]
    for name in grp.keys():
        print(f"  features[{k}][{name}] shape:", grp[name].shape)

f.close()
print("\nDone.")