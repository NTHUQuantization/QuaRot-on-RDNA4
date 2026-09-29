"""Reject mislabeled or incomplete GPTQ checkpoints before loading a runtime."""
import json
from pathlib import Path
import struct


def require_gptq_checkpoint(checkpoint):
    root=Path(checkpoint)
    config=json.loads((root/'config.json').read_text())
    conversion=config.get('quarot_conversion',{})
    if conversion.get('method')!='gptq' or conversion.get('w_bits')!=4:
        raise ValueError(f'Expected GPTQ INT4 conversion metadata: {root}')
    weight_map=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    shards=set(weight_map.values())
    if len(shards)!=config['num_hidden_layers']+1:
        raise ValueError(f'Incomplete streamed GPTQ shard set: {root}')
    for name in sorted(shards):
        path=root/name
        with path.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0]
            if not 0<length<100_000_000:
                raise ValueError(f'Invalid safetensors header: {path}')
            header=json.loads(f.read(length))
        signature=json.loads(header.get('__metadata__',{}).get('quarot_signature','{}'))
        if signature!=conversion:
            raise ValueError(f'GPTQ shard signature disagrees with config: {path}')
        keys=set(header)-{'__metadata__'}
        if keys!={k for k,v in weight_map.items() if v==name}:
            raise ValueError(f'GPTQ index/shard tensor mismatch: {path}')
        if max(header[k]['data_offsets'][1] for k in keys)+8+length!=path.stat().st_size:
            raise ValueError(f'Truncated GPTQ shard: {path}')
    return conversion
