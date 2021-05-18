import pandas as pd
import pickle
from collections import OrderedDict

def history_to_df(pickle_pth):
    with open(pickle_pth, "rb") as fh:
        h = pickle.load(fh)
    
    n_metric= len(h.tuning_history[0]['baseline'])
    dflist = []
    for episode, episode_history in enumerate(h.tuning_history[0]['history']):
        episode_cfg = []
        for layer_id, (k, v) in enumerate(episode_history['tune_cfg']['op'].items()):
            layercfg = OrderedDict()
            if len(k) == 3:
                layer_name, layer_type, m = k
            elif len(k) == 2:
                layer_name, layer_type = k
                
            layercfg['episode'] = episode
            layercfg['layer_id'] = layer_id
            layercfg['layer_name'] = layer_name
            layercfg['layer_type'] = layer_type
            
            if len(k) > 2:
                layercfg['m'] = m

            if 'weight' in v:
                for kw,vw in v['weight'].items():
                    layercfg['w_'+kw]=vw
            else:
                pass

            if 'activation' in v:
                for ka,va in v['activation'].items():
                    layercfg['a_'+ka]=va
            else:
                pass
            
            for i in range(0, n_metric):
                layercfg['tune_res'+str(i)] = episode_history['tune_result'][i]
            episode_cfg.append(layercfg)
        dflist.append(pd.DataFrame.from_dict(episode_cfg))

    history_df = pd.concat(dflist).reset_index(drop=True)
    return history_df