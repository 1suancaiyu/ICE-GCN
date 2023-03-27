# ICE-GCN
This repo is the official implementation for [ICE-GCN: An interactional channel excitation enhanced graph convolutional network
for skeleton-based action recognition]. The paper is accepted to Machine Vision and Applications.

An interactional channel excited graph convolutional network with complementary topology (ICE-GCN) is proposed. Extensive experiments and ablation studies demonstrate the necessity of the ICE module and the complementary topology scheme. Compared with previous works, the main contributions of our work can be summarized as follows:

• Compared with the existing attention mechanisms which ignore the crossdimensional interaction, our interactional channel excitation (ICE) module
embeds spatio-temporal information into channel attention, which allows to
explore discriminative spatio-temporal features of actions in a finer channel
level, adaptively recalibrating spatial-temporal-aware attention maps along
channel dimension. ICE, composed of a channel-wise temporal excitation
(CTE) and a channel-wise spatial excitation (CSE), can be inserted into any
existing graph convolutional networks as a plug-and-play module to enhance
the performances notably without light computational cost.
• We systematically investigate the strategies of graphs and argue that complementary topology is necessary. Three adjacency sub-matrices Ap, Al and
As are combined to construct the graph topology. This simple but efficient scheme notably improves the performance, which solves the dilemma
between adaptation and too large of a searching space.
• Finally, together equipped with ICE, an interactional channel excited
graph convolutional network with complementary topology (ICE-GCN) is
proposed, extensive experiments conducted on three large-scale datasets,
NTU RGB+D 60, NTU RGB+D 120, and Kinetics-Skeleton demonstrate
our ICE-GCN outperforms the state-of-the-art performance. The follow-up
ablation experiments and visualization also show the effectiveness of the
individual modules in graph convolutional networks.

## Acknowledgements

This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN).
Thanks to the original authors for their work!

# Contact
For any questions, feel free to contact: `wgsuxi@gmail.com`
