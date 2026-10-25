#!/usr/bin/env python3
"""Converts Real-ESRGAN SRVGGNetCompact checkpoints (.pth) to the F16
safetensors H3cApp bundles (RealESRGANScaler in VideoUpscale.swift).

No PyTorch needed: a .pth is a zip of a pickle plus raw tensor storages,
and the pickle is read with an unpickler that only builds tensors and
ordered dicts (anything else in it is refused).

  python3 -E tools/convert_realesrgan.py general  realesr-general-x4v3.pth realesr-general-wdn-x4v3.pth out.safetensors
  python3 -E tools/convert_realesrgan.py anime    realesr-animevideov3.pth out.safetensors
  python3 -E tools/convert_realesrgan.py rrdb 2    RealESRGAN_x2plus.pth out.safetensors
  python3 -E tools/convert_realesrgan.py rrdb 4    RealESRGAN_x4plus.pth out.safetensors

"general" blends the two checkpoints as inference_realesrgan.py does by
default (denoise strength 0.5: 0.5 * general + 0.5 * wdn, per tensor).
Weights: https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause,
Copyright (c) 2021, Xintao Wang).
"""
import collections, json, pickle, struct, sys, zipfile
import numpy as np

_DTYPES = {'FloatStorage': np.float32, 'HalfStorage': np.float16, 'BFloat16Storage': None}


class _Storage:
    def __init__(self, dtype, data):
        self.dtype, self.data = dtype, data


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=None, hooks=None, metadata=None):
    array = np.frombuffer(storage.data, dtype=storage.dtype)
    if not size:
        return array[offset:offset + 1].reshape(())
    item = array.itemsize
    return np.lib.stride_tricks.as_strided(array[offset:], shape=tuple(size),
                                           strides=tuple(s * item for s in stride)).copy()


def load_pth(path):
    archive = zipfile.ZipFile(path)
    names = archive.namelist()
    root = names[0].split('/')[0]
    pkl = archive.read(f'{root}/data.pkl')

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                return _rebuild_tensor_v2
            if module == 'collections' and name == 'OrderedDict':
                return collections.OrderedDict
            if module == 'torch' and name in _DTYPES:
                return name
            raise pickle.UnpicklingError(f'refusing {module}.{name}')

        def persistent_load(self, pid):
            kind, storage_type, key, _location, _numel = pid
            if kind != 'storage' or _DTYPES.get(storage_type) is None:
                raise pickle.UnpicklingError(f'unsupported storage {pid}')
            return _Storage(_DTYPES[storage_type], archive.read(f'{root}/data/{key}'))

    state = Restricted(__import__('io').BytesIO(pkl)).load()
    for key in ('params_ema', 'params'):
        if key in state:
            return state[key]
    return state


def save_safetensors(path, tensors, metadata):
    header, offset, blobs = {'__metadata__': metadata}, 0, []
    for name, array in tensors.items():
        data = np.ascontiguousarray(array.astype(np.float16)).tobytes()
        header[name] = {'dtype': 'F16', 'shape': list(array.shape), 'data_offsets': [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    with open(path, 'wb') as out:
        out.write(struct.pack('<Q', len(encoded)) + encoded + b''.join(blobs))


def main():
    kind = sys.argv[1]
    if kind == 'general':
        a, b = load_pth(sys.argv[2]), load_pth(sys.argv[3])
        tensors = {k: 0.5 * a[k].astype(np.float32) + 0.5 * b[k].astype(np.float32) for k in a}
        out, num_conv = sys.argv[4], 32
    elif kind == 'anime':
        tensors = {k: v.astype(np.float32) for k, v in load_pth(sys.argv[2]).items()}
        out, num_conv = sys.argv[3], 16
    elif kind == 'rrdb':
        # RRDBNet (RealESRGAN_x2plus / x4plus): 23 RRDB blocks, kept as they are.
        scale = sys.argv[2]
        tensors = {k: v.astype(np.float32) for k, v in load_pth(sys.argv[3]).items()}
        blocks = len({k.split('.')[1] for k in tensors if k.startswith('body.')})
        save_safetensors(sys.argv[4], tensors, {'arch': 'RRDBNet', 'num_feat': '64', 'num_grow_ch': '32',
                                                'num_block': str(blocks), 'scale': scale,
                                                'source': 'https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause)'})
        print('wrote', sys.argv[4], len(tensors), 'tensors,', blocks, 'blocks, conv_first', tensors['conv_first.weight'].shape)
        return
    else:
        raise SystemExit('kind must be general, anime or rrdb')
    convs = [k for k in tensors if k.endswith('.weight') and tensors[k].ndim == 4]
    assert len(convs) == num_conv + 2, (len(convs), num_conv)
    save_safetensors(out, tensors, {'arch': 'SRVGGNetCompact', 'num_feat': '64', 'num_conv': str(num_conv),
                                    'upscale': '4', 'act_type': 'prelu',
                                    'source': 'https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause)'})
    for k in sorted(tensors, key=lambda k: int(k.split('.')[1]))[:4]:
        print(k, tensors[k].shape)
    print('wrote', out, len(tensors), 'tensors')


if __name__ == '__main__':
    main()
