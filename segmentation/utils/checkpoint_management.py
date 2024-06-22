

def determine_weight_model(state_dict):
    if 'state_dict' in state_dict:
        state_dict_extracted = state_dict['state_dict']

        # ODIN1 test
        for k in state_dict_extracted:
            if k.startswith('online_network'):
                return 'odin1'
            elif k.startswith('network.encoder'):
                return 'odin2'
    else:
        # Leopart test
        if 'center' in state_dict:
            return 'leopart'
        # SSL4EO test
        if 'student' in state_dict and 'teacher' in state_dict:
            return 'ssl4eo'
        for k in state_dict:
            if k.startswith('online_network'):
                return 'odin1'
            elif k.startswith('network.encoder'):
                return 'odin2'

        return 'unknown'

def prepare_weights_for_vit_adapt(state_dict, model_key):
    # print(model_key)
    # print(state_dict.keys())
    if model_key == 'odin1':
        rel_key = 'online_network.encoder.'
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {
            k
            # .replace(rel_key, '')
            # .replace('.proj.', '.projection.')
            # .replace('blocks.', 'layers.')
            # .replace('.norm', '.ln')
            # .replace('blocks.', 'layers.')
            # .replace('.mlp.fc1.', '.ffn.layers.0.0.')
            # .replace('.mlp.fc2.', '.ffn.layers.1.')
            # .replace('.attn.qkv.', '.attn.attn.in_proj_')
            # .replace('.attn.projection.', '.attn.attn.out_proj.')
            # .replace('norm.', 'ln1.')
            : v for k, v in state_dict.items() if k.startswith(rel_key)
        }
    elif model_key == 'odin2':
        rel_key = 'network.encoder.'
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {
            k
            # .replace(rel_key, '')
            # .replace('.proj.', '.projection.')
            # .replace('blocks.', 'layers.')
            # .replace('.norm', '.ln')
            # .replace('blocks.', 'layers.')
            # .replace('.mlp.fc1.', '.ffn.layers.0.0.')
            # .replace('.mlp.fc2.', '.ffn.layers.1.')
            # .replace('.attn.qkv.', '.attn.attn.in_proj_')
            # .replace('.attn.projection.', '.attn.attn.out_proj.')
            # .replace('norm.', 'ln1.')
            : v for k, v in state_dict.items() if k.startswith(rel_key)
        }
    elif model_key == 'leopart':
        state_dict = {
            k
            # .replace('backbone.', '')
            # .replace('.proj.', '.projection.')
            # .replace('blocks.', 'layers.')
            # .replace('.norm', '.ln')
            # .replace('blocks.', 'layers.')
            # .replace('.mlp.fc1.', '.ffn.layers.0.0.')
            # .replace('.mlp.fc2.', '.ffn.layers.1.')
            # .replace('.attn.qkv.', '.attn.attn.in_proj_')
            # .replace('.attn.projection.', '.attn.attn.out_proj.')
            # .replace('norm.', 'ln1.')
            : v for k, v in state_dict.items()
        }
    elif model_key == 'ssl4eo':
        state_dict = {
            k
            # .replace('backbone.', '')
            # .replace('.proj.', '.projection.')
            # .replace('blocks.', 'layers.')
            # .replace('.norm', '.ln')
            # .replace('blocks.', 'layers.')
            # .replace('.mlp.fc1.', '.ffn.layers.0.0.')
            # .replace('.mlp.fc2.', '.ffn.layers.1.')
            # .replace('.attn.qkv.', '.attn.attn.in_proj_')
            # .replace('.attn.projection.', '.attn.attn.out_proj.')
            # .replace('norm.', 'ln1.')
            : v for k, v in state_dict['teacher'].items()
        }
    else:
        state_dict = {
            k
            # .replace('backbone.', '')
            # .replace('.proj.', '.projection.')
            # .replace('blocks.', 'layers.')
            # .replace('.norm', '.ln')
            # .replace('blocks.', 'layers.')
            # .replace('.mlp.fc1.', '.ffn.layers.0.0.')
            # .replace('.mlp.fc2.', '.ffn.layers.1.')
            # .replace('.attn.qkv.', '.attn.attn.in_proj_')
            # .replace('.attn.projection.', '.attn.attn.out_proj.')
            # .replace('norm.', 'ln1.')
            : v for k, v in state_dict.items()
        }

    return state_dict