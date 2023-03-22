import sys
from ptflops import get_model_complexity_info
from torchstat import stat
from thop import profile
import torch

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

if __name__ == "__main__":
    ice_gcn = import_class('model.ICE_GCN.Model')
    ctrgcn = import_class('model.ctrgcn.Model')
    agcn = import_class('model.agcn.Model')
    aagcn = import_class('model.aagcn.Model')
    agcn_stc = import_class('model.agcn_stc_attention.Model')
    agcn_ice = import_class('model.agcn_ICE.Model')
    ctrgcn_baseline = import_class('model.baseline.Model')

    model_chose = "agcn"
    # model_chose = "ctrgcn"
    # model_chose = "ice_gcn"
    # model_chose = "ctrgcn_baseline"

    if model_chose == "agcn" or "aagcn" or "agcn_stc" or "agcn_ice":
        T = 300
    else:
        T =64

    N = 64
    V = 25
    C = 3
    M = 2

    model = agcn(
        num_class = 60,
        num_point = 25,
        num_person = 2,
        graph = 'graph.ntu_rgb_d.Graph',
        graph_args = {'labeling_mode': 'spatial'}
    )

    macs, params = get_model_complexity_info(model, (C, T, V, M), as_strings=True, print_per_layer_stat=True, verbose=True)
    print("Model: ", model_chose)
    print("Input tensor shape:", C, T, V, M)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    inputs = torch.randn(1, C, T, V, M)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops)
    print('params: ', params)