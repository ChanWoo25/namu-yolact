## https://github.com/dbolya/yolact/tree/master/external/DCNv2
## Deformable Convolutional Networks V2 with Pytorch 1.0

### Build
```bash
    ./make.sh         # build
    python test.py    # run examples and gradient check 
```

### An Example
- deformable conv
```python
    from dcn_v2 import DCN
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    # torch.randn() : Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)
    # 함수에 인자로 들어온 수가 tensor의 size를 결정한다.
    ## 4차원 일듯. 128x128x64 이 2개
    # .cuda() : method를 통해 tensor를 GPU로 보낸다.
    dcn = DCN(64, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2).cuda()
    output = dcn(input)
    print(output.shape)
```
- deformable roi pooling
```python
    from dcn_v2 import DCNPooling
    input = torch.randn(2, 32, 64, 64).cuda()
    batch_inds = torch.randint(2, (20, 1)).cuda().float()
    x = torch.randint(256, (20, 1)).cuda().float()
    y = torch.randint(256, (20, 1)).cuda().float()
    w = torch.randint(64, (20, 1)).cuda().float()
    h = torch.randint(64, (20, 1)).cuda().float()
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)
    # torch.randint(low=0, high, size,,, : low(포함)와 high(불포함) 사이의 무작위 integer들로 채워진 tensor를 반환

    # torch.cat(tensors, dim=0, out=None) : 주어진 tensor들을 입력받은 차원(dim)으로 이어 붙인다.
    # >>> x = torch.randn(2, 3)
    # >>> x
    # tensor([[ 0.6580, -1.0969, -0.4614],
    #         [-0.1034, -0.5790,  0.1497]])
    # >>> torch.cat((x, x, x), 0)
    # tensor([[ 0.6580, -1.0969, -0.4614],
    #         [-0.1034, -0.5790,  0.1497],
    #         [ 0.6580, -1.0969, -0.4614],
    #         [-0.1034, -0.5790,  0.1497],
    #         [ 0.6580, -1.0969, -0.4614],
    #         [-0.1034, -0.5790,  0.1497]])
    # >>> torch.cat((x, x, x), 1)
    # tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
    #          -1.0969, -0.4614],
    #         [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
    #          -0.5790,  0.1497]])


    # mdformable pooling (V2)
    # wrap all things (offset and mask) in DCNPooling
    dpooling = DCNPooling(spatial_scale=1.0 / 4,
                         pooled_size=7,
                         output_dim=32,
                         no_trans=False,
                         group_size=1,
                         trans_std=0.1).cuda()

    dout = dpooling(input, rois)
```
### Note
Now the master branch is for pytorch 1.0 (new ATen API), you can switch back to pytorch 0.4 with,
```bash
git checkout pytorch_0.4
```

### Known Issues:

- [x] Gradient check w.r.t offset (solved)
- [ ] Backward is not reentrant (minor)

This is an adaption of the official [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets/tree/master/DCNv2_op).

<s>I have ran the gradient check for many times with DOUBLE type. Every tensor **except offset** passes.
However, when I set the offset to 0.5, it passes. I'm still wondering what cause this problem. Is it because some
non-differential points? </s>

Update: all gradient check passes with double precision. 

Another issue is that it raises `RuntimeError: Backward is not reentrant`. However, the error is very small (`<1e-7` for 
float `<1e-15` for double), 
so it may not be a serious problem (?)

Please post an issue or PR if you have any comments.
    
