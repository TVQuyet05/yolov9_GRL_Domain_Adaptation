import torch

def transfer_weights(old_weights_path, new_weights_path):
    # Load old checkpoint
    ckpt = torch.load(old_weights_path, map_location=\'cpu\')

    # Get state_dict
    state_dict = ckpt.get(\'model\', ckpt)
    if hasattr(state_dict, \'state_dict\'):
        state_dict = state_dict.state_dict()
    elif type(state_dict) is dict and \'state_dict\' in state_dict:
        state_dict = state_dict[\'state_dict\']
    elif type(ckpt) is dict and \'model_state_dict\' in ckpt:
        state_dict = ckpt[\'model_state_dict\']
        
    if type(state_dict) is not dict and hasattr(state_dict, \'half\'):
        state_dict = state_dict.half().state_dict() # for fp16 models

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(\'model.\'):
            parts = k.split(\'.\')
            try:
                idx = int(parts[1])
                # Shift indices after CPCA insertion
                if idx >= 10:
                    idx += 1
                
                parts[1] = str(idx)
                new_k = \'.\'.join(parts)
                new_state_dict[new_k] = v
            except ValueError:
                new_state_dict[k] = v
        else:
            new_state_dict[k] = v

    # Modify the checkpoint and save
    if type(ckpt) is dict and \'model\' in ckpt:
        # replace just the state dict inside the model object if applicable
        # to ensure safe resuming, usually saving state_dict directly is safest
        ckpt[\'model\'] = new_state_dict
        torch.save(ckpt, new_weights_path)
    else:
        torch.save(new_state_dict, new_weights_path)
        
    print(f\'Successfully transferred network weights to {new_weights_path}\')
    print(\'You can now load this new checkpoint. (New modules will be initialized randomly due to strict=False during training load)\')

if __name__ == \'__main__\':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(\'--old\', type=str, default=\'yolov9-s.pt\', help=\'old weights\')
    parser.add_argument(\'--new\', type=str, default=\'yolov9s_cpca_ready.pt\', help=\'new weights\')
    args = parser.parse_args()
    transfer_weights(args.old, args.new)
