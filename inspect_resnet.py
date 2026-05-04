import h5py

f = h5py.File(r'backend/brain_tumor_resnet50_fulltune.h5', 'r')

print('=== FC ===')
for k in f['weights']['fc'].keys():
    item = f['weights']['fc'][k]
    if hasattr(item, 'shape'):
        print(f'  [{k}] shape: {item.shape}')



    else:
        for k2 in item.keys():
            print(f'  [{k}][{k2}] shape: {item[k2].shape}')



print('=== CONV1 ===')
for k in f['weights']['conv1'].keys():
    item = f['weights']['conv1'][k]
    if hasattr(item, 'shape'):
        print(f'  [{k}] shape: {item.shape}')
    else:
        for k2 in item.keys():
            print(f'  [{k}][{k2}] shape: {item[k2].shape}')

print('=== LAYER1 (first key only) ===')
first = list(f['weights']['layer1'].keys())[0]
def print_group(g, prefix=''):
    for k in g.keys():
        item = g[k]
        if hasattr(item, 'shape'):
            print(f'  {prefix}[{k}] shape: {item.shape}')
        else:
            print_group(item, prefix+f'[{k}]')
print_group(f['weights']['layer1'][first], f'layer1[{first}]')

f.close()

<<<<<<< HEAD
print('Done.')
=======
print('Done.')


>>>>>>> 1854273c38c7f10f3bd36904a32aa2a566058d52
