[*UNDER CONSTRUCTION*] 

# ABC-Net-Pytorch
My implemenation of [ABC-Net](https://arxiv.org/abs/1711.11294). Currently I finish the ABC-Net on alexnet architecture on imagenet dataset, but the model does NOT converge. I am considering use some small dataset which is relative easy to debug.

***IF YOU FIND THE MISTAKEs I MADE IN THE IMPLEMNATION THAT CAUSE THE DISCONVERGENCY, PLEASE LET ME KNOW. thanks: )***

## TO-DO
- ABC-Net in resnet-18 architecture on cifar10 dataset

## Dismatchs
Considering some details are NOT specified in the paper, I make modifications as follow:
- *the way to solve alpha.*: The lstsq(scipy.linalg.lstsq) method is adopted.
- *STE attach to each Binary Base OR full precision Weight*: fp Weight
- *gradient of W(see following Notes)*: keep as the original paper
- *alexnet architecture*: basicly adopting the modification in Xnor-Net excepting kernel size to keep identity with pretrained model.


## Notes
I find some preoblem of this paper reported in the [notes](https://github.com/cow8/ABC-Net-pytorch/raw/master/notes.pdf).
